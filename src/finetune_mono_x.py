import argparse

import numpy as np
import torch

from pathlib import Path

from torch import nn
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm import tqdm

from pytorch_loader import IRDatasetLoader

# from batch_sampler import UnbalancedClassesSampler

MODEL_MAPPING = {
    "bert": "bert-base-uncased",
    "roberta": "roberta-base",
    "electra": "google/electra-base-discriminator"
}


def main():
    parser = argparse.ArgumentParser(description="Fine tune monox model")
    parser.add_argument("llm", type=str, help="""
                                                Large Language Model (LLM) to be used.
                                                Possible choice:
                                                - bert
                                                - roberta
                                                - electra
                                                """)
    parser.add_argument("train_dataset",
                        type=str,
                        help="Train dataset to use from ir dataset, default")

    parser.add_argument("train_pp_ds", type=str, help="Path to train pre-processed dataset")
    parser.add_argument('--continue_training', action='store_true', help="If presente, continue training")

    parser.add_argument('--tokenizer', type=str, help="Tokenizer to use. Only useful if continue_training is True")
    parser.add_argument("--batch_size", type=int, help="Batch size to use, default 32", default=32)
    parser.add_argument("--device", type=str, help="Device to use", default="cuda")

    parser.add_argument("--output_dir", type=str, help="Default as the same string as llm argument", default=None)
    parser.add_argument("--save_after", type=int, help="Number of training steps to wait before saving the checkpoint",
                        default=1000)
    parser.add_argument("--num_epochs", type=int, help="Number of epochs", default=1)
    parser.add_argument("--num_training_steps", type=int, help="Number of training steps", default=None)

    args = parser.parse_args()

    # Validate arguments
    batch_size = args.batch_size

    print(f"{batch_size=}")

    num_epochs = args.num_epochs

    print(f"{num_epochs=}")

    num_training_steps = int(1e5)

    print(f"{num_training_steps=}")

    train_dataset = args.train_dataset
    train_pp_ds = args.train_pp_ds

    if args.output_dir is None:
        output_dir = Path(args.llm)
    else:
        output_dir = Path(args.output_dir)

    if args.llm not in MODEL_MAPPING and not args.continue_training:
        raise Exception(f"LLM passed via --lm argument non under consideration. Value passed: {args.llm}")

    if not args.continue_training:
        model_path = MODEL_MAPPING[args.llm]  # hugging face large language model name
    else:
        model_path = args.llm

    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    # Load tokenizer and datasets
    device = torch.device(args.device)
    if not args.continue_training:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_MAPPING[args.tokenizer])

    train_dataset = IRDatasetLoader(train_dataset, tokenizer, irdataset_pt_file_name=train_pp_ds)

    gen = torch.Generator()
    gen.manual_seed(2147483647)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=gen)
    
    # for param in model.base_model.parameters():
    #    param.requires_grad = False

    optimizer = AdamW(model.parameters(), lr=1e-6)

    num_warmup_steps = 1e4

    print(f"{num_warmup_steps=}")

    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    model.to(device)

    model.train()

    step_n = 0
    window_loss = []
    for epoch in range(num_epochs):
        with tqdm(train_dataloader, unit="batch", total=min(num_training_steps, len(train_dataloader))) as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                batch_forward = {k: v.to(device) for k, v in batch.items() if k not in ["query_id", "doc_id", "labels"]}
                outputs = model(**batch_forward)
                logits = outputs.logits
                cel_w = nn.CrossEntropyLoss().to(device)
                loss_w = cel_w(logits.to(device), batch["labels"].to(device))
                loss_w.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                window_loss.append(loss_w.item())
                if len(window_loss) < 100:
                    tepoch.set_postfix(loss="---")
                else:
                    tepoch.set_postfix(loss=np.mean(window_loss))
                    window_loss = window_loss[1:]
                step_n += 1
                if step_n % args.save_after == 0:
                    output_dir_step = output_dir/str(step_n)
                    print(f"Saving checkpoint in: {output_dir_step}")
                    model.save_pretrained(output_dir_step)

                if step_n == num_training_steps:
                    break

    output_dir_step = output_dir / str(step_n)
    model.save_pretrained(output_dir_step)


if __name__ == '__main__':
    main()
