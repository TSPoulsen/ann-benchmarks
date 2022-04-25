from __future__ import absolute_import
import puffinn
from ann_benchmarks.algorithms.base import BaseANN
import numpy

class PuffinnPQ(BaseANN):
    def __init__(self, metric, params): #, space=10**6, hash_function="", hash_source='pool', hash_args=None):

        if metric not in ['jaccard', 'angular']:
            raise NotImplementedError(
                    "Puffinn doesn't support metric %s" % metric)
        self.metric = metric
        self.space =  params.get("space",10**6)
        self.hash_function = params.get("hash_function", "fht_crosspolytope")
        self.hash_source = params.get("hash_source", "pool")
        self.hash_args = params.get("hash_args", None) 
        self.use_pq = params.get("use_pq", True)
        self.M = params.get("M", 8)
        self.K = params.get("K", 256)
        self.loss = params.get("loss", "euclidean")

    def fit(self, X):
        if self.metric == 'angular':
            dimensions = len(X[0])
        else:
            dimensions = 0
            for x in X:
                dimensions = max(dimensions, max(x)+1)

        if self.hash_args:
            self.index = puffinn.Index(self.metric, dimensions, self.space,\
                    hash_function=self.hash_function, hash_source=self.hash_source,\
                    hash_args=self.hash_args,
                    use_pq=self.use_pq, M=self.use_pq, K=self.use_pq, loss=self.loss)
        else:
            self.index = puffinn.Index(self.metric, dimensions, self.space,\
                    hash_function=self.hash_function, hash_source=self.hash_source,
                    use_pq=self.use_pq, M=self.use_pq, K=self.use_pq, loss=self.loss)
        for i, x in enumerate(X):
            x = x.tolist()
            self.index.insert(x)
        self.index.rebuild()

    def set_query_arguments(self, recall):
        self.recall = recall

    def query(self, v, n):
        v = v.tolist()
        return self.index.search(v, n, self.recall)

    def __str__(self):
        if (self.use_pq):
            return 'PUFFINNPQ(space=%d, recall=%f, hf=%s, hashsource=%s, M=%s, K=%s, loss=%s)' % (self.space, self.recall, self.hash_function, self.hash_source, self.M, self.K, self.loss)
        else:
            return 'PUFFINNPQ(space=%d, recall=%f, hf=%s, hashsource=%s)' % (self.space, self.recall, self.hash_function, self.hash_source)

