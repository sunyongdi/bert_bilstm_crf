import torch
from model import BertNer
from data_loader import get_data
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizer, BertConfig
from trainer import Trainer
import os
import config

args = config.Args().get_parser()
from loguru import logger

logger.add('runtime.log')

import wandb
wandb.login()

wandb.init(project="ner",name='bert_bilstm_crf', config=args)



def build_optimizer_and_scheduler(args, model, t_total):
    module = (
        model.module if hasattr(model, "module") else model
    )

    # 差分学习率
    no_decay = ["bias", "LayerNorm.weight"]
    model_param = list(module.named_parameters())

    bert_param_optimizer = []
    other_param_optimizer = []

    for name, para in model_param:
        space = name.split('.')
        # print(name)
        if space[0] == 'bert_module' or space[0] == "bert":
            bert_param_optimizer.append((name, para))
        else:
            other_param_optimizer.append((name, para))

    optimizer_grouped_parameters = [
        # bert other module
        {"params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': args.bert_learning_rate},
        {"params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': args.bert_learning_rate},

        # 其他模块，差分学习率
        {"params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': args.crf_learning_rate},
        {"params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': args.crf_learning_rate},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.bert_learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(args.warmup_proportion * t_total), num_training_steps=t_total
    )

    return optimizer, scheduler


def main():
    wandb.config = args
    tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = os.path.join(args.data_dir, args.data_name, "ner_data")
    with open(os.path.join(data_path, "labels.txt"), "r", encoding="utf-8") as fp:
        labels = fp.read().strip().split("\n")
        
    bio_labels = ["O"]
    for label in labels:
        bio_labels.append("B-{}".format(label))
        bio_labels.append("I-{}".format(label))

    
    label2id = {label: i for i, label in enumerate(bio_labels)}
    id2label = {i: label for i, label in enumerate(bio_labels)}
    train_loader, dev_loader = get_data(data_path, tokenizer, args.max_seq_len, label2id, args.train_batch_size, args.dev_batch_size)
    t_toal = len(train_loader) * args.epochs
    
    # model config
    model_config = BertConfig.from_pretrained(args.bert_dir)
    model_config.num_labels = len(bio_labels)
    model_config.max_seq_len = args.max_seq_len
    model_config.create_name = 'syd'
    model_config.label2id = label2id
    model_config.id2label = id2label
    model = BertNer(model_config)

    wandb.watch(model)
    model.to(device)
    
    optimizer, schedule = build_optimizer_and_scheduler(args, model, t_toal)
    train = Trainer(
        output_dir=args.output_dir,
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=dev_loader,
        optimizer=optimizer,
        schedule=schedule,
        epochs=args.epochs,
        device=device,
        save_step=args.save_step,
        id2label=id2label
    )

    train.train()
    model = BertNer.from_pretrained(os.path.join(args.output_dir, 'ner'))
    report = train.test(model)
    logger.info(report)
    tokenizer.save_pretrained(os.path.join(os.path.join(args.output_dir, 'ner')))    


if __name__ == "__main__":
    data_name = "dgre"
    main()
