import dataset
import json
import logging
import os
import hashlib

logger = logging.getLogger(__name__)

class open_table():

    def __init__(self, path, table_name):
        self.path = path
        self.table_name = table_name
        return None;

    def __enter__(self):
        self.db = dataset.connect('sqlite:///%s' % self.path)
        return self.db[self.table_name]

    def __exit__(self, type, value, traceback):
        self.db.executable.close()


class ResultSet(object):

    def __init__(self, db, process_ctx):
        self.db = db
        self.path = db.replace('series.db', '')
        self.process_ctx = process_ctx
        self.filters = None
        self.cache = []
        self.queried = [] # store recent queries
        self.includes = None # if this set is defined they are used as SELECT instead of * to reduce query time
        self.params = [] # all columns that start with param_
        self.jsonized = []
        # newer versions use statistics.db instead of series.db
        self.db_statistics = db.replace('series.db', 'statistics.db')
        if os.path.exists(self.db_statistics):
            logger.info('using statistics.db instead of series.db')
        else:
            self.db_statistics = None

    def get(self, name):
        runs = list(filter(self._run_filter, data))

        pass

    def print_info(self):
        """Print details about this result set (also called a blob) """
        def pline(key, value):
            print(key.ljust(10), str(value))   

        print("~~~~~~~~~ <blob_info> ~~~~~~~~~")
        pline("db", self.db)
        pline("db_size", os.path.getsize(self.db))


        if os.path.exists(self.db):
            # add details about the content of the database
            db = dataset.connect('sqlite:///%s' % self.db)
            # print the tables that are present in the db
            pline("db_tables", ', '.join(db.tables))  
            #for table in db.tables:
            #    pline(table, table.count('*'))

        if self.db_statistics:
            if os.path.exists(self.db_statistics):
                pline("db_statistics", self.db_statistics)
                pline("db_statistics_size", os.path.getsize(self.db_statistics))
                db = dataset.connect('sqlite:///%s' % self.db_statistics)

                keys = []
                dts_result = 0
                for column in db['statistics'].columns:
                    if column.startswith("param"): continue
                    if 'dts_result' in column: 
                        dts_result += 1
                        continue;
                    if column.startswith("json_"):
                        column = column.replace('json_', '')
                    keys.append(column)
                for k in sorted(keys):
                    pline(k, "")

        print("~~~~~~~~~ </blob_info> ~~~~~~~~~")
        
    def filterProcessCtx(self, **kwargs):
        print("filterProcessCtx", kwargs)

        search = dict()
        for k, v in kwargs.items():
            search[k.replace('.', '_')] = v

        resultSet = []
        for item in self.process_ctx:
            useItem = True
            for k, v in search.items():
                if item.get(k) != v:
                    useItem = False
            if useItem:
                resultSet.append(item)
        return resultSet

    def filterCache(self, **kwargs):
        self.filters = kwargs
        return list(filter(self._run_filter, self.cache))


    def find_columns(self, keyword):
        result = []
        with open_table(self.db_statistics, 'statistics') as table:
            for c in table.columns:
                if keyword in c:
                    if c.startswith('json_'):
                        c = c.replace('json_', '')
                    result.append(c)
        return result

    def include_parameters(self, **kwargs):
        """Usually all items in the table are selected; however, the table can have
        several hundret columns so it might be useful to restrict the SELECT statement from
        * to the set of parameters that are actually required. This set is defined here."""
        self.includes = []
        for k,v in kwargs.items():
            if v==1:
                self.includes.append(k)
   
        
    def group_by(self, **kwargs):
        """The group_by operator takes a single parameter name with a list of
        parameter values, i.e. something like param_xyz=[v1, v2, ...]. It will then
        go through all the runs in the selected database and group the runs by the 
        specified values. The result is a list of arrays where the each element looks like
        [r_v1, r_v2, ...], that is, all parameters except for param_xyz are identical for
        all elements, only the value of param_xyz is different."""
        if len(kwargs) != 1:
            raise RuntimeError('group_by operators supports exactly one kwargs parameter!')
        filter_values = []
        result_map = {}
        for k, values in kwargs.items():
            #print("handle k", k)
            for v in values:
                if not v in filter_values:
                    filter_values.append(v)
                f = dict()
                f[k] = v
                #print("filter", f)
                runs = self.filter(**f)
                for r in runs:
                    # we need a unique key for this run that includes all
                    # parameters except the one that was given in kwargs
                    params = {}
                    for x in r.keys():
                        if x.startswith('param_'):
                            if x != k:
                                params[x] = r.get(x)
                    key = (str(sorted(params.items())))

                    if not result_map.get(key):
                        result_map[key] = {}
                    result_map[key][v]  = r

        #print(filter_values)
        #print(len(result_map))
        result = []
        for key, runs in result_map.items():
            if len(runs) == len(filter_values):
                new_result = []
                for v in filter_values:
                    new_result.append(runs.get(v))
                result.append(new_result)        
        return result

    def _run_filter(self, el):
        if not self.filters: return True;
        for k, v in self.filters.items():
            if isinstance(v, list):
                pass
            else:
                if el.get(k.replace('.', '_')) != v:
                    return False
        return True

    def run_filter(self, database, **kwargs):
        all_columns = []
        if len(self.params) == 0:
            with open_table(database, 'statistics') as table:
                print("open")
                for c in table.columns:
                    all_columns.append(c)

                    if c.startswith('param_'):
                        self.params.append(c)
                    if c.startswith('json_'):
                        self.jsonized.append(c)


        # check includes for columns that are jsonized (lists are stored 
        # in a single column as a json formatted string)
        if self.includes:
            for i, c in enumerate(self.includes):
                if c not in all_columns and 'json_'+c in self.jsonized:
                    self.includes[i] = 'json_' + c

        where_clause = []
        for k, v in kwargs.items():
            where_clause.append('%s = %d\n' % (k.replace('.', '_'), v))
        select = '*'
        if self.includes:
            select = ','.join(self.params + self.includes)

        if len(where_clause) > 0:
            qry = '''
                SELECT %s
                FROM statistics
                WHERE
                %s;''' % (select, ' AND '.join(where_clause))
        else:
            qry = '''
                SELECT %s
                FROM statistics;''' % (select)      

        logger.info("------------- query -----------")
        logger.info(qry)
        db = dataset.connect('sqlite:///%s' % database)
        runs = []
        result = db.query(qry)
        processed = 0
        for entry in result:
            r = dict(entry)
            for k, v in entry.items():
                if k.startswith('json_'):
                    data = None
                    if v:
                        data = json.loads(v)
                    r[k.replace('json_', '')] = data
                    del r[k]
            runs.append(r)
            processed +=1
        logger.info("processed entries: %d" % processed)
        logger.info("------------- query done -----------")
        return list(filter(self._run_filter, runs))

    def run_filter_db(self, database, **kwargs):
        filterkey = str(list(sorted(kwargs.items())))
        dbsize = '%d' % os.path.getsize(database)
        uuid_str = filterkey+dbsize
        include_str = ''
        if self.includes:
            include_str = str(list(sorted(self.includes)))
        m = hashlib.md5()      
        m.update(uuid_str.encode('utf-8'))  
        m.update(include_str.encode('utf-8'))

        uuid = str(m.hexdigest())
        #print(uuid)

        cache = os.path.join(self.path, '.cache')
        if not os.path.exists(cache):
            os.makedirs(cache)
        cachefile = os.path.join(cache, uuid)
        """
        do_not_cache = self.ctx.config.get('param_do_not_cache', -1) # flag that says that no cache file is created

        if self.ctx.config.get('param_delete_cache') and self.ctx.config.get('param_delete_cache') > 0:
            if os.path.exists(cachefile):
                logger.info("delete cached file for switch=%s because param_delete_cache > 0" % switch.label)
                os.remove(cachefile)
        """
        if os.path.exists(cachefile):
            #print("cache found")
            with open(cachefile, 'r') as file:
                result = json.loads(file.read())
                return result
        else:
            print("no cache, run")
            #raise RuntimeError()
            result = self.run_filter(database, **kwargs)
            with open(cachefile, 'w') as file:
                file.write(json.dumps(result))           
            return result


    def filter(self, **kwargs):
        if self.db_statistics:
            return self.run_filter_db(self.db_statistics, **kwargs)
        else:
            self.filters = kwargs
            # older versions used series.db directly and the parameters where stored
            # in a different table (results instead of statistics) so that a join is required
            where_clause = ''
            search = dict()
            for k, v in kwargs.items():
                search[k.replace('.', '_')] = v
                where_clause += '            AND results.%s = %d\n' % (k.replace('.', '_'), v)

            qry = '''
                SELECT results.*, statistics.*
                FROM results
                INNER JOIN statistics
                ON results.id = statistics.result_id
                %s;''' % where_clause

            if not qry in self.queried:
                self.queried.append(qry)
                #print("------------- query -----------")
                #print(qry)
                db = dataset.connect('sqlite:///%s' % self.db)
                result = db.query(qry)
                #self.cache = []
                for entry in result:
                    r = dict(entry)
                    for k, v in entry.items():
                        if k.startswith('json_'):
                            data = None
                            if v:
                                data = json.loads(v)
                            r[k.replace('json_', '')] = data
                            del r[k]

                    self.cache.append(r)
            return list(filter(self._run_filter, self.cache))
        self.filters = kwargs

        # ---------- old
        if self.db_statistics:
            # newer version where statistics.db does exist

            if len(self.params) == 0:
                with open_table(self.db_statistics, 'statistics') as table:
                    for c in table.columns:
                        if c.startswith('param_'):
                            self.params.append(c)

            where_clause = []
            search = dict()
            for k, v in kwargs.items():
                search[k.replace('.', '_')] = v
                where_clause.append('%s = %d\n' % (k.replace('.', '_'), v))

            select = '*'
            if self.includes:
                select = ','.join(self.params + self.includes)
            qry = '''
                SELECT %s
                FROM statistics
                WHERE
                %s;''' % (select, ' AND '.join(where_clause))

            if not qry in self.queried:
                self.queried.append(qry)
                print("------------- query -----------")
                print(qry)
                db = dataset.connect('sqlite:///%s' % self.db_statistics)
                result = db.query(qry)
                for entry in result:
                    r = dict(entry)
                    for k, v in entry.items():
                        if k.startswith('json_'):
                            data = None
                            if v:
                                data = json.loads(v)
                            r[k.replace('json_', '')] = data
                            del r[k]

                    self.cache.append(r)

    
            return list(filter(self._run_filter, self.cache))

        else:
            # older versions used series.db directly and the parameters where stored
            # in a different table (results instead of statistics) so that a join is required
            where_clause = ''
            search = dict()
            for k, v in kwargs.items():
                search[k.replace('.', '_')] = v
                where_clause += '            AND results.%s = %d\n' % (k.replace('.', '_'), v)

            qry = '''
                SELECT results.*, statistics.*
                FROM results
                INNER JOIN statistics
                ON results.id = statistics.result_id
                %s;''' % where_clause

            if not qry in self.queried:
                self.queried.append(qry)
                #print("------------- query -----------")
                #print(qry)
                db = dataset.connect('sqlite:///%s' % self.db)
                result = db.query(qry)
                for entry in result:
                    r = dict(entry)
                    for k, v in entry.items():
                        if k.startswith('json_'):
                            data = None
                            if v:
                                data = json.loads(v)
                            r[k.replace('json_', '')] = data
                            del r[k]

                    self.cache.append(r)
            return list(filter(self._run_filter, self.cache))
