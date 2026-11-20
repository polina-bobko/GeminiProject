"""HW10: LoRA fine-tuning a small causal LM to generate QA bug reports."""
import json
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import random
import re
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUT = Path("outputs_tinyllama_v4")
OUT.mkdir(exist_ok=True)
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# Each item is a unique synthetic QA scenario (not repeated rows).
SCENARIOS = {
    "UI": [
        ("Login button overlaps the password field on mobile", "Login button overlaps password field", "Login form elements do not overlap", "Button covers part of the password field"),
        ("Profile avatar is stretched on the settings page", "Profile avatar is distorted", "Avatar keeps its aspect ratio", "Avatar appears stretched"),
        ("Checkout heading is cut off on narrow screens", "Checkout heading is clipped", "Checkout heading is fully visible", "Part of the heading is cut off"),
        ("Tooltip appears behind the navigation menu", "Tooltip is hidden behind menu", "Tooltip appears above other elements", "Navigation menu covers tooltip"),
        ("Dark mode text is unreadable on the payment screen", "Payment text has low contrast", "Text remains readable in dark mode", "Text blends into the background"),
        ("Search icon is missing from the header", "Search icon is not displayed", "Search icon is visible in header", "Header shows no search icon"),
        ("Modal dialog extends beyond the viewport", "Modal exceeds screen boundaries", "Modal fits inside the viewport", "Bottom of modal is outside screen"),
        ("Product cards have inconsistent spacing", "Product card spacing is inconsistent", "Product cards use consistent spacing", "Gaps between cards are uneven"),
        ("Error message is displayed outside the form", "Form error message is misplaced", "Validation message appears near its field", "Error message appears outside form"),
        ("Footer links overlap on tablet screens", "Footer links overlap on tablet", "Footer links remain separated", "Footer links overlap"),
        ("Notification badge is partially hidden", "Notification badge is clipped", "Notification badge is fully visible", "Part of notification badge is hidden"),
        ("Dropdown arrow is misaligned with its label", "Dropdown arrow is misaligned", "Arrow aligns with dropdown label", "Arrow is shifted below the label"),
        ("Loading spinner is not centered in the dialog", "Loading spinner is off-center", "Spinner is centered in dialog", "Spinner appears near dialog edge"),
        ("Sidebar text wraps over adjacent icons", "Sidebar labels overlap icons", "Labels do not overlap icons", "Wrapped labels cover icons"),
        ("Date picker calendar is hidden under the footer", "Date picker is obscured", "Calendar appears above footer", "Footer hides calendar rows"),
        ("Breadcrumbs are truncated without an ellipsis", "Breadcrumb truncation is unclear", "Truncated breadcrumbs show ellipsis", "Breadcrumb text ends abruptly"),
        ("Primary button uses the wrong background color", "Primary button color is incorrect", "Button uses design-system color", "Button has a different color"),
        ("Checkbox label is not aligned with the checkbox", "Checkbox label is misaligned", "Label aligns with checkbox", "Label appears below checkbox"),
        ("Toast message is clipped at the right edge", "Toast message is clipped", "Toast text is fully visible", "Right side of toast is cut off"),
        ("Table header overlaps the first data row", "Table header overlaps row", "Table header does not overlap data", "Header covers first row"),
    ],
    "FUNCTIONAL": [
        ("Login fails with valid credentials", "Login fails with valid credentials", "Valid credentials allow login", "Login request fails"),
        ("Reset password email is never sent", "Password reset email is missing", "Reset email is sent", "No reset email arrives"),
        ("Cart total does not update after removing an item", "Cart total is not recalculated", "Total updates after item removal", "Old total remains visible"),
        ("Search returns no results for an existing product", "Search misses existing product", "Existing products appear in results", "Search returns an empty list"),
        ("User can submit registration form with an invalid email", "Registration accepts invalid email", "Invalid email is rejected", "Form accepts invalid email"),
        ("Logout button does not end the user session", "Logout does not clear session", "Logout ends the session", "User remains authenticated"),
        ("Order confirmation is not shown after payment", "Order confirmation is missing", "Successful payment shows confirmation", "Confirmation page does not appear"),
        ("Profile changes disappear after refreshing the page", "Profile edits are not saved", "Profile changes persist", "Changes disappear after refresh"),
        ("Download invoice button opens a blank page", "Invoice download fails", "Invoice file is downloaded", "Blank page opens instead"),
        ("Filter by category includes unrelated products", "Category filter returns wrong items", "Filter returns only matching products", "Unrelated products appear"),
        ("User can add an out-of-stock item to the cart", "Unavailable item can be added", "Out-of-stock items cannot be added", "Item is added despite no stock"),
        ("Language selection resets after navigation", "Selected language is not retained", "Language preference persists", "Language returns to default"),
        ("Save address button does not persist the address", "Shipping address is not saved", "Saved address persists", "Address is lost after navigation"),
        ("Pagination skips the last search result", "Pagination omits a result", "All search results are accessible", "Last result never appears"),
        ("Two-factor code is accepted after expiration", "Expired verification code is accepted", "Expired codes are rejected", "Expired code allows access"),
        ("Cancel subscription button keeps subscription active", "Subscription cancellation fails", "Cancellation deactivates subscription", "Subscription remains active"),
        ("Duplicate payment is created after double-clicking Pay", "Payment is submitted twice", "Only one payment is processed", "Two charges are created"),
        ("Comment edit action deletes the comment", "Editing comment removes it", "Editing updates existing comment", "Comment disappears"),
        ("Favorite button does not save selected products", "Favorites are not saved", "Selected products remain in favorites", "Favorites list remains empty"),
        ("Invite link leads to an invalid team page", "Team invitation link is broken", "Invite link opens team onboarding", "Invalid team page appears"),
    ],
    "PERFORMANCE": [
        ("Dashboard takes 12 seconds to load", "Dashboard loads slowly", "Dashboard loads within acceptable time", "Dashboard takes 12 seconds"),
        ("Product search freezes for several seconds", "Product search freezes", "Search responds without freezing", "Search freezes during execution"),
        ("Scrolling through the catalog is very laggy", "Catalog scrolling is laggy", "Catalog scrolls smoothly", "Scrolling stutters"),
        ("Report export takes more than a minute", "Report export is slow", "Report export completes promptly", "Export takes over one minute"),
        ("Opening a large chat history makes the app unresponsive", "Chat history causes UI freeze", "Chat history opens responsively", "App stops responding"),
        ("Image gallery takes too long to display thumbnails", "Gallery thumbnails load slowly", "Thumbnails appear promptly", "Thumbnails are delayed"),
        ("Typing in the search field has a noticeable delay", "Search input has high latency", "Text input responds immediately", "Characters appear with delay"),
        ("Checkout page slows down when cart has many items", "Checkout slows for large carts", "Checkout remains responsive", "Large carts cause delays"),
        ("Memory usage increases continuously during navigation", "Navigation causes memory growth", "Memory usage remains stable", "Memory consumption keeps increasing"),
        ("API response time exceeds ten seconds at peak load", "API latency spikes at peak load", "API responds within target latency", "Response exceeds ten seconds"),
        ("Switching tabs causes repeated UI freezes", "Tab switching freezes UI", "Tab changes are responsive", "UI freezes on tab changes"),
        ("Autocomplete suggestions appear after a long pause", "Autocomplete is delayed", "Suggestions appear quickly", "Suggestions arrive after a pause"),
        ("Loading user profile blocks the interface", "Profile loading blocks UI", "Interface stays responsive during loading", "Interface is blocked"),
        ("Uploading a small attachment takes unusually long", "Small file upload is slow", "Small attachments upload promptly", "Upload takes excessive time"),
        ("Opening settings triggers a long loading spinner", "Settings page loads slowly", "Settings opens promptly", "Long spinner appears"),
        ("Table sorting stalls on large datasets", "Large table sorting stalls", "Sorting finishes responsively", "Sorting stalls"),
        ("Switching between photos causes frame drops", "Photo navigation drops frames", "Photo navigation stays smooth", "Animation stutters"),
        ("App startup is much slower after an update", "App startup regressed", "App starts within expected time", "Startup takes longer"),
        ("Map zooming becomes sluggish with many markers", "Map zoom is sluggish", "Map zoom remains responsive", "Zooming lags"),
        ("Notification list takes several seconds to open", "Notifications open slowly", "Notification list opens promptly", "Opening list is delayed"),
    ],
}


def build_dataframe():
    rows = []
    for category, cases in SCENARIOS.items():
        for description, title, expected, actual in cases:
            rows.append({"description": description, "title": title, "category": category,
                         "expected": expected, "actual": actual})
    df = pd.DataFrame(rows)
    for column in df.columns:
        df[column] = df[column].str.replace(r"\s+", " ", regex=True).str.strip()
    df = df.dropna().drop_duplicates(subset=["description"]).reset_index(drop=True)
    return df


FIELDS = ("Category", "Title", "Expected", "Actual")
TASKS = ("Category", "Title", "Behavior")


def prompt_for(description, task):
    if task == "Behavior":
        return (
            "Write exactly two lines for this QA bug report. "
            "Expected means the CORRECT behavior that should happen. "
            "Actual means the INCORRECT behavior observed in the bug. "
            "Do not swap them. Start the lines with Expected: and Actual:. "
            f"Issue: {description}"
        )
    if task == "Category":
        return f"Classify this software bug as UI, FUNCTIONAL, or PERFORMANCE. Reply with only the category. Issue: {description}"
    return f"Write a short, factual bug report title describing what is broken. Reply with only the title. Issue: {description}"


def task_answer(row, task):
    if task == "Behavior":
        return f"Expected: {row['expected']}\nActual: {row['actual']}"
    return str(row[task.lower()])


def augment_description(description):
    # Augment only within the training split; keep validation/test unseen.
    return [description, f"Bug observed: {description}", f"Problem: {description}"]


def examples_for(df, augment=False):
    examples = []
    for _, row in df.iterrows():
        descriptions = augment_description(row["description"]) if augment else [row["description"]]
        for desc in descriptions:
            for task in TASKS:
                examples.append({"description": desc, "task": task, "answer": task_answer(row, task)})
    return examples


def chat_prompt(tokenizer, description, task):
    messages = [{"role": "user", "content": prompt_for(description, task)}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def tokenize_row(row, tokenizer):
    prefix = chat_prompt(tokenizer, row["description"], row["task"])
    prompt_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(row["answer"] + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    ids = (prompt_ids + answer_ids)[:256]
    labels = ([-100] * len(prompt_ids) + answer_ids)[:256]
    if len(prompt_ids) + len(answer_ids) > 256:
        raise ValueError("Example exceeds max length; increase token budget")
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


class PaddingCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, examples):
        max_len = max(len(x["input_ids"]) for x in examples)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for ex in examples:
            pad = max_len - len(ex["input_ids"])
            batch["input_ids"].append(ex["input_ids"] + [self.pad_id] * pad)
            batch["attention_mask"].append(ex["attention_mask"] + [0] * pad)
            batch["labels"].append(ex["labels"] + [-100] * pad)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def generate_task(model, tokenizer, description, task, device):
    model.eval()
    inputs = tokenizer(chat_prompt(tokenizer, description, task), return_tensors="pt").to(device)
    with torch.no_grad():
        result = model.generate(
            **inputs, max_new_tokens=90 if task == "Behavior" else 40,
            do_sample=False, repetition_penalty=1.08,
            pad_token_id=tokenizer.eos_token_id,
        )
    continuation = result[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(continuation, skip_special_tokens=True).strip()


def parse_behavior(text):
    # Never silently exchange fields: preserve model output and flag invalid reports.
    expected = re.search(r"(?im)^Expected:\s*(.+)$", text)
    actual = re.search(r"(?im)^Actual:\s*(.+)$", text)
    return (expected.group(1).strip() if expected else "",
            actual.group(1).strip() if actual else "")


def generate_report(model, tokenizer, description, device):
    category = generate_task(model, tokenizer, description, "Category", device).splitlines()[0].strip()
    title = generate_task(model, tokenizer, description, "Title", device).splitlines()[0].strip()
    behavior = generate_task(model, tokenizer, description, "Behavior", device)
    expected, actual = parse_behavior(behavior)
    return {"Category": category, "Title": title, "Expected": expected, "Actual": actual}


def normalized_category(value):
    match = re.match(r"^(UI|FUNCTIONAL|PERFORMANCE)\b", value.strip(), re.IGNORECASE)
    return match.group(1).upper() if match else "UNKNOWN"


def behavior_quality(report):
    # This is a format/duplication check, NOT a semantic correctness score.
    exp = report["Expected"].strip().lower()
    act = report["Actual"].strip().lower()
    return bool(exp and act and exp != act)

def field_value(row, field):
        field_to_column = {
            "Category": "category",
            "Title": "title",
            "Expected": "expected",
            "Actual": "actual",
        }

        column = field_to_column[field]
        return str(row[column])


def main():
    df = build_dataframe()
    assert len(df) == 60
    df.to_csv(OUT / "dataset_clean.csv", index=False)
    train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["category"])
    train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=SEED, stratify=train_val_df["category"])
    for name, split in (("train", train_df), ("validation", val_df), ("test", test_df)):
        split.to_csv(OUT / f"{name}.csv", index=False)
    print(f"Original scenarios: {len(df)}; train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).to(device)
    model.config.use_cache = False  # Required for gradient checkpointing during training
    model.config.pad_token_id = tokenizer.pad_token_id

    print("Evaluating baseline...")
    baseline = [generate_report(model, tokenizer, row["description"], device) for _, row in test_df.iterrows()]

    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16,
        lora_dropout=0.05, target_modules=["q_proj", "v_proj"],
    ))
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    train_examples = examples_for(train_df, augment=True)
    val_examples = examples_for(val_df)
    print(f"Task examples: train={len(train_examples)}, validation={len(val_examples)}")
    train_ds = Dataset.from_list(train_examples)
    val_ds = Dataset.from_list(val_examples)
    train_ds = train_ds.map(lambda row: tokenize_row(row, tokenizer), remove_columns=train_ds.column_names)
    val_ds = val_ds.map(lambda row: tokenize_row(row, tokenizer), remove_columns=val_ds.column_names)

    args = TrainingArguments(
        output_dir=str(OUT / "checkpoints"), per_device_train_batch_size=1,
        per_device_eval_batch_size=1, gradient_accumulation_steps=8, num_train_epochs=3, learning_rate=2e-4,
        weight_decay=0.01, logging_steps=20, evaluation_strategy="epoch",
        save_strategy="no", report_to="none", fp16=False, bf16=False,
        seed=SEED, dataloader_num_workers=0, use_cpu=(device == "cpu"),
        gradient_checkpointing=True, optim="adamw_torch",
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=PaddingCollator(tokenizer.pad_token_id))
    trainer.train()
    validation_metrics = trainer.evaluate()
    model.config.use_cache = True
    model.save_pretrained(OUT / "fine_tuned_adapter")
    tokenizer.save_pretrained(OUT / "fine_tuned_adapter")

    predictions = []
    print("Generating test reports...")
    for (_, row), before in zip(test_df.iterrows(), baseline):
        after = generate_report(model, tokenizer, row["description"], device)
        predictions.append({
            "description": row["description"], "expected_category": row["category"],
            **{f"before_{field.lower()}": before[field] for field in FIELDS},
            **{f"after_{field.lower()}": after[field] for field in FIELDS},
            **{f"reference_{field.lower()}": field_value(row, field) for field in FIELDS},
        })
    results = pd.DataFrame(predictions)
    results.to_csv(OUT / "test_predictions.csv", index=False)
    before_categories = results["before_category"].map(normalized_category)
    after_categories = results["after_category"].map(normalized_category)
    expected = results["expected_category"]
    metrics = {
        "test_count": len(results),
        "validation_loss": float(validation_metrics["eval_loss"]),
        "category_accuracy_before": float((before_categories == expected).mean()),
        "category_accuracy_after": float((after_categories == expected).mean()),
        "complete_fields_before": float(results[[f"before_{f.lower()}" for f in FIELDS]].apply(lambda r: all(bool(str(x).strip()) for x in r), axis=1).mean()),
        "complete_fields_after": float(results[[f"after_{f.lower()}" for f in FIELDS]].apply(lambda r: all(bool(str(x).strip()) for x in r), axis=1).mean()),
        "behavior_distinct_before": float(sum(behavior_quality(r) for r in baseline) / len(baseline)),
        "behavior_distinct_after": float(sum(behavior_quality({f: row[f"after_{f.lower()}"] for f in FIELDS}) for _, row in results.iterrows()) / len(results)),
        "note": "Expected/Actual are now generated together. Presence and distinctness do not prove semantic correctness: review test_predictions.csv manually.",
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("\nCLASSIFICATION REPORT AFTER FINE-TUNING:")
    print(classification_report(expected, after_categories, labels=["UI", "FUNCTIONAL", "PERFORMANCE"], zero_division=0))
    print("\nTEST RESULTS:\n", json.dumps(metrics, indent=2))
    for _, sample in results.head(3).iterrows():
        print("\nINPUT:", sample["description"])
        print("REFERENCE CATEGORY:", sample["expected_category"])
        print("BEFORE:", {f: sample[f"before_{f.lower()}"] for f in FIELDS})
        print("AFTER:", {f: sample[f"after_{f.lower()}"] for f in FIELDS})
    print("\nSaved to:", OUT.resolve())


if __name__ == "__main__":
    main()
