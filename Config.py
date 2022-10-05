class Configration(object):
    def __init__(self, batch_size=500, epochs=500, lr=0.003, seed=17, log_interval=100, data='FB15K-237', l2=0.0, model='conve', embedding_dim=200, embedding_shape1=20,
                hidden_drop=0.3, input_drop=0.2, feat_drop=0.2, lr_decay=0.995, loader_threads=4, preprocess=True, use_bias=True, label_smoothing=0.1, hidden_size=4608):
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.log_interval = log_interval
        self.data = data
        self.l2 = l2
        self.model = model
        self.embedding_dim = embedding_dim
        self.embedding_shape1 = embedding_shape1
        self.hidden_drop = hidden_drop
        self.input_drop = input_drop
        self.feat_drop = feat_drop
        self.lr_decay = lr_decay
        self.loader_threads = loader_threads
        self.preprocess = preprocess
        self.use_bias = use_bias
        self.label_smoothing = label_smoothing
        self.hidden_size = hidden_size