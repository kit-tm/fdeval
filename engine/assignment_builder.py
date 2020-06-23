import logging
import itertools
from functools import reduce 
import operator
import time

logger = logging.getLogger(__name__)

# examples (debugging only!)
demand = [0, 3, 3, 3, 5, 6, 5, 6, 4, 4, 4]
jobs = [6, 5, 5, 4, 3, 3, 2, 2, 2, 2, 2]
demand_all = [19, 19, 20, 16, 12, 15, 11, 12, 10, 11, 10]
s1 = [0, -9, -9, -9, -9, -9, -9, 9, 4, -9, 4]
s2 = [9, 9, 7, 7, -9, -9, -9, -9, -9, -9, 8]

#demand =  [3,  3,  3,  5,  6,  5,  6,  4,  4 ]
#s1 =      [5,  7,  4,  8,  10, 14, 19, 14, 10]
#s2 =      [19, 17, 17, 14, 0,  0,  15, 11, 11]
#jobs =    [4,  4,  3,  2,  2,  2,  2,  2,  2 ]
#demand_all =  [12, 14, 12, 8, 8, 7, 9, 8, 9]

#demand =  [1, 2, 3, 3, 4, 5, 5, 4, 3, 1, 1, 3, 5, 5, 6, 5, 5, 5, 4, 4, 6, 5, 5, 4, 5, 5, 6, 6, 5, 4, 4, 5, 4, 4, 3, 4]
#jobs =  [3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
#demand_all =  [5, 3, 7, 8, 9, 11, 11, 8, 7, 5, 5, 7, 12, 13, 14, 13, 12, 11, 9, 4, 6, 5, 5, 4, 5, 5, 6, 6, 5, 4, 4, 5, 4, 4, 3, 4]
#s1 =  [32, 31, 33, 36, 35, 28, 26, 29, 27, 26, 31, 31, 32, 35, 32, 35, 34, 34, 30, 27, 27, 34, 32, 29, 28, 32, 31, 33, 37, 38, 38, 35, 30, 27, 22, 21]
#s2 =  [16, 22, 23, 24, 25, 23, 25, 30, 28, 25, 27, 27, 28, 31, 30, 31, 25, 25, 23, 32, 37, 37, 39, 38, 36, 33, 29, 29, 30, 32, 31, 27, 29, 30, 34, 33]

class SubAssignment():

    def __init__(self, left_index, right_index, rating, table_demand, swfree):
        self.li = left_index
        self.ri = right_index
        self.rating = rating
        self.table_demand = table_demand
        self.swfree = swfree
        # make sure that table_demand was created with correct dimensions
        assert(right_index-left_index == len(table_demand)-1)
        # make sure that the switch dict was created with correct dimensions
        first_in_swfree = next(iter(swfree.values()))
        assert(len(table_demand) == len(first_in_swfree))
        self.switch_options = self.get_partial()

    def info(self):
        logger.error(str([' %2.d ' % x for x in self.table_demand]))
        for sw, swfree in self.swfree.items():
            logger.error(str([' %2.d ' % x for x in swfree])+ " <-" +str(sw))
        pass

    def get_partial(self, include_backup_switch = False):
        """return the suitable remote switch options for this partial assignment"""
        result = []
        if include_backup_switch:
            result = [['ES'] * len(self.table_demand)]    
        for sw, swfree in self.swfree.items():
            # calculate the difference between free capacity in switch sw
            # and the capacity demand for the corresponding time slot which is
            # stored in table_demand; add these differences to an array and select
            # the minimum; if it is >=0, the switch is available in all time slots
            if min([a-b for a,b in zip(swfree, self.table_demand)]) >= 0:
                result.append([sw]*len(self.table_demand))
        return result

    def split(self):

        ratings = [0] * len(self.table_demand)  

        for i, _ in enumerate(ratings):
            need = self.table_demand[i]
            cnt_fit = 0
            for sw, swfree in self.swfree.items():
                free = swfree[i]
                ratings[i] += free-need
                if free >= need: cnt_fit+=1
            if cnt_fit > 1:
                ratings[i] *= cnt_fit

        #print("ratings:", ratings, sum(ratings))
        bestcut = -10000000000
        bestcut_val = None
        cuts = []
        length = len(self.table_demand)
        for cut, d in enumerate(self.table_demand):
            if cut == 0: continue;
            # left branch
            r_left = min(ratings[0:cut])*cut
            r_right = min(ratings[cut:])*(length-cut)

            cut_rating = r_left+ r_right
            if cut_rating > bestcut:
                bestcut = cut_rating
                bestcut_val = ((0, cut-1), (cut, length-1), r_left, r_right)
            cuts.append((r_left, r_right, cut_rating))

        cut_left, cut_right, rating_left, rating_right = bestcut_val

        swfree_left = {}
        swfree_right = {}
        for sw, swfree in self.swfree.items():
            swfree_left[sw] = swfree[:cut_left[1]+1]
            swfree_right[sw] = swfree[cut_right[0]:]

        left = SubAssignment(cut_left[0], cut_left[1], rating_left, 
            self.table_demand[:cut_left[1]+1], swfree_left)

        right = SubAssignment(cut_right[0], cut_right[1], rating_right,
            self.table_demand[cut_right[0]:], swfree_right)


        return left, right

class AssignmentBuilder():
    
    def __init__(self, verbose=False, max_assignments=80):
        self.max_assignments = max_assignments
        self.verbose = verbose
        self.swfree = {}
        self.table_demand = None
        self.sub_assignments = []

    def add_demand(self, data):
        self.table_demand = data

    def add_switch(self, sw, data):
        self.swfree[sw] = data

    def finalize(self):

        # first re-create the assignments with the backup switch added to
        # the switch options
        options = []
        for a in self.sub_assignments:
            # overwrite the switch list
            a.switch_options = a.get_partial(include_backup_switch=True)
            options.append(a.switch_options)

        cart = list(itertools.product(*options))
        # flatten the result that was returned by itertools.product()
        plain_options = []
        for c in cart:
            option = []
            for e in c:
                option += e
            if not option in plain_options:
                plain_options.append(option)

        #print("  result (%d options):" % len(plain_options))
        if self.verbose: 
            for option in plain_options:
                #print("    ->", option)
                pass

        return plain_options     

    def run_recursive(self, lvl):
        if lvl >= 15:
            #print("  ==> max level %d reached, finalize and return" % lvl)
            return self.finalize()  

        #print("#"*20)
        #print("#### lvl = %d assignment_parts=%d" % (
        #    lvl, len(self.sub_assignments)))
        #print("#"*20)

        # 1) add all valid assignments for the current level; in addition, calculate
        # the number of assignments that would be created by the cartesian power over
        # all partial assignments; important: for scalability reasons, the cartesian power
        # operation is only executed as soon as the threshold for the number of assignments
        # was hit and the function is (almost) finished
        #print("---------> add assignments for this level")
        estimate_result_cnt = []
        for a in self.sub_assignments:
            # we add the number of the elemnts in this assignment to the
            # helper array and we add +1 for the backup switch which is not
            # included in the initial problem (for scalability)
            estimate_result_cnt.append(len(a.switch_options)+1)
        
        estimate_cnt = reduce(operator.mul, estimate_result_cnt, 1)

        # abort condition
        if estimate_cnt > self.max_assignments:
            #print("  ==> %d > %d; finalize and return" % (estimate_cnt, self.max_assignments))
            return self.finalize()

        options = []
        for a in self.sub_assignments:
            options.append(a.switch_options)
        for i, part in enumerate(options):
            #print("  assignment options for part", i, part)
            pass

        # 2) select the sub assignment with the lowest rating for split()
        # higher rating means the prob. to select this chunk as a whole is higher while
        # lower rating means that most likely only the backup switch is a suitable option
        #print("---------> select split option")
        valid_options = []
        for a in self.sub_assignments:
            # we cannot split an assignment with length=1 (obviously)
            if len(a.table_demand) > 1:
                #print("  r=%d" % a.rating)
                valid_options.append(a) 

        if len(valid_options) == 0:
            #print("  ==> no more options finalize and return")
            return self.finalize()  

        split_candidate = min(valid_options, key=lambda x: x.rating)
        
        # 3) execute the split
        #print(" ** split %d" % split_candidate.rating)
        a, b = split_candidate.split()
        if self.verbose:
            #print("---------> left split; rating=%d options=%s" % (a.rating, str(a.switch_options)))
            a.info()
            #print("---------> right split; rating=%d options=%s" % (b.rating, str(b.switch_options)))
            b.info()

        # 4) now replace the one that was splitted with the two parts from the split
        # but keep the same position in the array, i.e. if [a,b,c,d] and c was splitted
        # into c1 and c2 the new array would be [a,b,c1,c2,d]; this is important because
        # the sub-assignments array is ordered
        self.sub_assignments = list(itertools.chain.from_iterable(
            (a, b) if item == split_candidate else (item, ) for item in self.sub_assignments))

        # 5) execute the function again until one of the exit conditions is met
        return self.run_recursive(lvl+1)

    def run(self):
        top_assignment = SubAssignment(0, 
            len(self.table_demand)-1, 0, self.table_demand[:], self.swfree)
        self.sub_assignments = [top_assignment]
        return self.run_recursive(1)


if __name__== "__main__":
    t = time.time()
    ab = AssignmentBuilder(verbose=True)
    ab.add_switch('s1', s1)
    ab.add_switch('s2', s2)
    ab.add_demand(demand)

    ab.run()
    logger.error("time=%f" % (time.time()-t))
