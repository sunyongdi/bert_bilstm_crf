import os

from tqdm import tqdm
from seqeval.metrics import classification_report, f1_score

from loguru import logger
import wandb

class Trainer:
    def __init__(self,
                 output_dir=None,
                 model=None,
                 train_loader=None,
                 save_step=500,
                 dev_loader=None,
                 test_loader=None,
                 optimizer=None,
                 schedule=None,
                 epochs=1,
                 device="cpu",
                 id2label=None):
        self.output_dir = output_dir
        self.model = model
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.device = device
        self.optimizer = optimizer
        self.schedule = schedule
        self.id2label = id2label
        self.save_step = save_step
        self.total_step = len(self.train_loader) * self.epochs

    def train(self):
        global_step = 1
        best_f1 = 0
        for epoch in range(1, self.epochs + 1):
            for step, batch_data in enumerate(self.train_loader):
                self.model.train()
                for key, value in batch_data.items():
                    batch_data[key] = value.to(self.device)
                input_ids = batch_data["input_ids"]
                attention_mask = batch_data["attention_mask"]
                labels = batch_data["labels"]
                output = self.model(input_ids, attention_mask, labels)
                loss = output.loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.schedule.step()
                logger.info(f"【train】{epoch}/{self.epochs} {global_step}/{self.total_step} loss:{loss.item()}")
                wandb.log({'train_loss': loss.item()}, step=global_step)
                if global_step % self.save_step == 0:
                    dev_loss, f1_score = self.dev()
                    if best_f1 < f1_score:
                        best_f1 = f1_score
                        logger.info(f"dev{epoch}/{self.epochs} {global_step}/{self.total_step} loss:{dev_loss} best_f1:{best_f1}")
                        wandb.log({'dev_loss': dev_loss, 'best_f1': best_f1}, step=global_step)
                        self.model.save_pretrained(os.path.join(self.output_dir, 'ner'))
                global_step += 1

    def dev(self):
        self.model.eval()
        preds = []
        trues = []
        total_loss = 0
        num_batches = 0
        
        for step, batch_data in enumerate(tqdm(self.dev_loader)):
            for key, value in batch_data.items():
                batch_data[key] = value.to(self.device)
            input_ids = batch_data["input_ids"]
            attention_mask = batch_data["attention_mask"]
            labels = batch_data["labels"]
            output = self.model(input_ids, attention_mask, labels)
            logits = output.logits
            attention_mask = attention_mask.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            
            total_loss += output.loss.item()
            num_batches += 1
            
            batch_size = input_ids.size(0)
            for i in range(batch_size):
                length = sum(attention_mask[i])
                logit = logits[i][1:length]
                logit = [self.id2label[i] for i in logit]
                label = labels[i][1:length]
                label = [self.id2label[i] for i in label]
                preds.append(logit)
                trues.append(label)
                
        avg_loss = total_loss / num_batches
        f1 = f1_score(trues, preds)
        return avg_loss, f1
    
    
    def test(self, model):
        model.to(self.device)
        model.eval()
        preds = []
        trues = []
        for step, batch_data in enumerate(tqdm(self.test_loader)):
            for key, value in batch_data.items():
                batch_data[key] = value.to(self.device)
            input_ids = batch_data["input_ids"]
            attention_mask = batch_data["attention_mask"]
            labels = batch_data["labels"]
            output = model(input_ids, attention_mask, labels)
            logits = output.logits
            attention_mask = attention_mask.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            batch_size = input_ids.size(0)
            for i in range(batch_size):
                length = sum(attention_mask[i])
                logit = logits[i][1:length]
                logit = [self.id2label[i] for i in logit]
                label = labels[i][1:length]
                label = [self.id2label[i] for i in label]
                preds.append(logit)
                trues.append(label)
        report = classification_report(trues, preds)
        return report

