# Publish as matinfirooz/ConvForge-NumPy

Create an empty GitHub repository named **ConvForge-NumPy** under `matinfirooz`.
Leave GitHub's initial README, license, and .gitignore options unchecked; all
three are supplied in the package.

Inside the extracted project directory, with Git installed and your GitHub
account authenticated:

```bash
git init -b main
git add .
git commit -m "Build CNN from scratch with NumPy and a visual drawing laboratory"
git remote add origin https://github.com/matinfirooz/ConvForge-NumPy.git
git push -u origin main
```

Suggested repository description:

> CNN from scratch in NumPy: manual backpropagation, im2col convolution, learned filters, feature maps, pixel sensitivity, and an interactive drawing lab.

Suggested topics:

`numpy` `cnn` `convolutional-neural-network` `from-scratch` `deep-learning`
`computer-vision` `backpropagation` `im2col` `educational` `feature-visualization`

The README uses relative SVG paths and GitHub-compatible math. Curated results
and the small trained shape checkpoint are included. The CI matrix starts on
your first push; remote compatibility results are not claimed before it runs.

For GitHub's web uploader, upload the **contents** of the extracted directory,
placing README.md at the repository root. Include `.github` and `.gitignore`;
a Git push handles these automatically.

GitHub's HTML file viewer does not run the laboratory. Download `docs/cnn_lab.html`
to explore the static gallery; run `python -m examples.serve_lab` for live custom
drawings. All predictions in live mode are computed by the local NumPy server.
