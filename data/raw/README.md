# Raw data (not tracked in git)

This project uses **Fashion-MNIST** (Zalando), downloaded via `torchvision` into
this directory. The files are not committed.

Populate with:

```bash
make data   # python -m gan_robustness download
```

Fashion-MNIST: 60 000 training + 10 000 test grayscale images, 28×28, 10 balanced
classes of clothing (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt,
Sneaker, Bag, Ankle boot). See `docs/` and the report for the full description.
