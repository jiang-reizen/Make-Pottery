# Let's Make Pottery!

## Installation

1. Create a `Conda` environment

```bash
conda create --name MakePottery python=3.11
conda activate MakePottery
```

2. Install packages

```bash
pip install -r requirements.txt
cd py-vox-io
pip install .
```

3. Data

Extract the data into the `data/` directory. The final file structure should look like this:

```text
Make-Pottery/
  data/
    test/
    train/
  utils/
    FragmentDataset.py
    model_utils.py
    model.py
    utils.py
    visualize.py
  readme.md
  requirements.txt
  test.py
  training.py
  vis.ipynb
```

## Training

Run the following command for default training:

```bash
python training.py
```

You can also specify arguments such as the data path and checkpoint save location. Please refer to the source code for details.

The current training result is saved in `checkpoints/best_model_400.pth`.

## Testing

You can test the model using the following command:

```bash
python test.py --checkpoint [CHECKPOINT_PATH]
```

Sample Output:

```
Testing on device: cuda
[TEST] Dataset initialized. Found 4752 files.
Loaded checkpoint from checkpoints/best_model_400.pth
[Result] DSC: 0.7374 | IoU: 0.6058 | MSE: 0.0365
```

## Visualization

You can view the generation results in `vis.ipynb`. Use the class_index parameter to specify and view samples for a specific class.