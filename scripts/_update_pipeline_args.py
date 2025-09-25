from pathlib import Path
path = Path("polymer/training/achmra/pipeline.py")
text = path.read_text()
old_block = "        args_kwargs = dict(\n            output_dir=str(output_dir),\n            per_device_train_batch_size=phase.batch_size,\n            per_device_eval_batch_size=phase.batch_size,\n            gradient_accumulation_steps=phase.gradient_accumulation,\n            learning_rate=phase.learning_rate,\n            warmup_ratio=phase.warmup_ratio,\n            num_train_epochs=phase.epochs,\n            max_steps=phase.steps if phase.steps is not None else -1,\n            logging_steps=25,\n            eval_steps=100,\n            save_steps=200,\n            save_total_limit=2,\n            bf16=self.config.bf16,\n            report_to=[],\n        )"
if old_block not in text:
    raise SystemExit("block not found")
new_block = "        args_kwargs = dict(\n            output_dir=str(output_dir),\n            per_device_train_batch_size=phase.batch_size,\n            per_device_eval_batch_size=phase.batch_size,\n            gradient_accumulation_steps=phase.gradient_accumulation,\n            learning_rate=phase.learning_rate,\n            warmup_ratio=phase.warmup_ratio,\n            num_train_epochs=phase.epochs,\n            max_steps=phase.steps if phase.steps is not None else -1,\n            logging_steps=25,\n            eval_steps=100,\n            save_steps=200,\n            save_total_limit=2,\n            bf16=self.config.bf16,\n            report_to=[],\n            remove_unused_columns=False,\n        )"
text = text.replace(old_block, new_block)
path.write_text(text)
