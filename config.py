import argparse


class Args:
    @staticmethod
    def parse():
        parser = argparse.ArgumentParser()
        return parser


    @staticmethod
    def initialize(parser):
        parser.add_argument('--bert_dir', default='./model_hub/chinese-bert-wwm-ext/')
        parser.add_argument('--output_dir', default='./checkpoint/')
        parser.add_argument('--data_dir', default='./data/')
        parser.add_argument('--data_name', default='duie')
        parser.add_argument('--seed', type=int, default=123, help='random seed')
        parser.add_argument('--gpu_ids', type=str, default='0')
        parser.add_argument('--max_seq_len', default=256, type=int)
        parser.add_argument('--epochs', default=5, type=int)
        parser.add_argument('--train_batch_size', default=12, type=int)
        parser.add_argument('--dev_batch_size', default=12, type=int)
        parser.add_argument('--bert_learning_rate', default=3e-5, type=float)
        parser.add_argument('--crf_learning_rate', default=3e-3, type=float)
        parser.add_argument('--adam_epsilon', default=1e-8, type=float)
        parser.add_argument('--weight_decay', default=0.01, type=float)
        parser.add_argument('--warmup_proportion', default=0.01, type=float)
        parser.add_argument('--save_step', default=100, type=int)
        return parser

    def get_parser(self):
        parser = self.parse()
        parser = self.initialize(parser)
        return parser.parse_args()
    
if __name__ == "__main__":
    args = Args()
    print(args.get_parser())