## Pipeline

```mermaid
flowchart TD
    A["ElectroCom61 dataset<br/>2071 photos, YOLO detection format"] --> B["Crop each labeled box<br/>10 classes, 2320 single-object crops"]
    B --> C["Re-split by shooting session<br/>prevents near-duplicate leakage"]
    C --> D["train 1196 / val 507 / test 617<br/>224x224 images, batches of 32"]
    D --> E["MobileNetV2 base, frozen<br/>pretrained on ImageNet, 154 layers"]
    E --> F["GlobalAveragePooling2D + Dense 10<br/>only the new head trains, 12,810 params"]
    F --> G["Fine-tune last 30 base layers<br/>learning rate 0.00001"]
    G --> H["Evaluate once on test set<br/>93.35% accuracy, confusion matrix"]
    H --> I["Save .keras model + Gradio upload app"]
```

### Key decisions
- **Detection to classification:** the dataset has many components per photo, so each labeled box was cropped into its own image.
- **Session-based split:** the original train/valid/test folders contained photos taken seconds apart, so photos were grouped by capture session and each group kept in one split.
- **Transfer learning:** the pretrained base was frozen and only a new 10-class output layer was trained, then the last 30 base layers were unfrozen at a low learning rate.
- **Test set used once:** all decisions were made on validation data.

### Limitations
- Classifies one component per image. It is not an object detector.
- The model always picks one of its 10 classes, even for parts it was not trained on.
- The two IC base classes are the most confused pair.
