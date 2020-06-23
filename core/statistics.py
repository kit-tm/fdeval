import logging
import time

logger = logging.getLogger(__name__)

class Timer(object):
    """docstring for Timer"""
    def __init__(self, ctx, namespace):
        super(Timer, self).__init__()
        self.started = time.time()
        self.ctx = ctx
        self.namespace = namespace
        self.timers = []

    def start(self, key):
        t = time.time()
        self.timers.insert(0, (key, t))
        return self

    def stop(self):
        now = time.time()
        key, t = self.timers.pop()
        assert key, t
        logger.info('%s.%s : %.3fms' % (self.namespace, key, (now-t)*1000.0))
        self.ctx.statistics['timer.%s.%s' % (self.namespace, key)] = (now-t)*1000.0



        