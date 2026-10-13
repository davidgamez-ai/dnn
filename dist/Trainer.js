import { readdirSync } from "node:fs";
import { join } from "node:path";
import { AdamW } from "./AdamW.js";
import DEBUG from "./Debug.js";
import { ImageLoader } from "./ImageLoader.js";
import { Network } from "./Network.js";
import { random, setSeed } from "./Random.js";
/**
 * Trains a network on the digit images.
 */
export class Trainer {
    hyperparameters;
    _hyperparameters;
    _network;
    _optimizer;
    _imageLoader = new ImageLoader();
    _train = [];
    _validation = [];
    _test = [];
    /**
     * Builds the network, so its weights and biases are ready to use, and
     * creates the optimizer that will update them.
     *
     * @param hyperparameters Settings for training, including how to split the
     * data, the random seed and the optimizer's settings.
     * @param network The network to train. build() is called on it here, which
     * gives it fresh random weights.
     */
    constructor(hyperparameters, network) {
        this.hyperparameters = hyperparameters;
        this._hyperparameters = hyperparameters;
        this._network = network;
        this._network.build();
        this._optimizer = new AdamW(hyperparameters);
    }
    /**
     * Finds every image in the images folder and divides them into training,
     * validation and test sets, using the fractions in the hyperparameters'
     * dataSplit. Each image is labelled with its digit, which is the name of
     * the folder it is in. Calling it again replaces the previous sets.
     *
     * The images are shuffled before splitting. On disk they are grouped by
     * digit, so without shuffling the test set would contain only 8s and 9s.
     * The random number generator is first reset to the hyperparameters'
     * randomSeed, so the same seed always gives the same split.
     *
     * @param imagesDirectory Folder containing one subfolder per digit (0-9), each holding PNG images.
     */
    loadData(imagesDirectory = "data/images") {
        const images = [];
        for (const folder of readdirSync(imagesDirectory, { withFileTypes: true })) {
            if (!folder.isDirectory() || !/^[0-9]$/.test(folder.name)) {
                continue;
            }
            const digit = Number(folder.name);
            const folderPath = join(imagesDirectory, folder.name);
            for (const file of readdirSync(folderPath)) {
                if (file.toLowerCase().endsWith(".png")) {
                    images.push({ fileName: join(folderPath, file), digit });
                }
            }
        }
        // Sort first so the starting order does not depend on how the file system lists files.
        images.sort((a, b) => a.fileName.localeCompare(b.fileName));
        setSeed(this._hyperparameters.randomSeed);
        shuffle(images);
        const { train, validation } = this._hyperparameters.dataSplit;
        const trainCount = Math.round(images.length * train);
        const validationCount = Math.round(images.length * validation);
        // The test set takes whatever is left, so rounding never loses or duplicates an image.
        this._train = images.slice(0, trainCount);
        this._validation = images.slice(trainCount, trainCount + validationCount);
        this._test = images.slice(trainCount + validationCount);
        console.log(`Training images: ${this._train.length}`);
        console.log(`Validation images: ${this._validation.length}`);
        console.log(`Test images: ${this._test.length}`);
    }
    /**
     * Trains the network for the number of epochs in the hyperparameters. An
     * epoch is one complete pass through the training images.
     *
     * At the start of each epoch the training images are shuffled, so the
     * batches contain a different mix of images every time. Otherwise the
     * network would see the images in the same order each epoch, which can
     * make it learn patterns in the order rather than in the images.
     *
     * Within an epoch the images are processed in batches of batchSize. The
     * last batch may be smaller if the images do not divide evenly.
     *
     * For each image:
     *   1. Load and flatten the image, and feed it forward through the network.
     *   2. Measure the cross-entropy loss against the image's digit.
     *   3. Backpropagate, so every layer has its errors for this image.
     *   4. Calculate the gradients, averaging them over the images in the batch.
     *
     * At the end of each batch:
     *   5. The AdamW optimizer updates every layer's weights and biases using
     *      the batch's averaged gradients.
     *
     * Each batch's average loss is logged when DEBUG.BATCH_LOSS is set. After
     * each epoch, the epoch's average training loss and the loss on the
     * validation images (see calculateValidationLoss()) are logged. As the
     * network learns, the losses fall.
     *
     * @throws Error if loadData() has not been called, or found no training images.
     */
    train() {
        if (this._train.length === 0) {
            throw new Error("No training images; call loadData() first");
        }
        const epochs = this._hyperparameters.epochs;
        const batchSize = this._hyperparameters.batchSize;
        const numBatches = Math.ceil(this._train.length / batchSize);
        for (let epoch = 1; epoch <= epochs; epoch++) {
            // A new random order of training images for this epoch.
            shuffle(this._train);
            let epochLossTotal = 0;
            let batchCounter = 0;
            let batchLossTotal = 0;
            let batchNumber = 0;
            for (const image of this._train) {
                // Step 1: forward pass.
                const pixels = this._imageLoader.flatten(this._imageLoader.load(image.fileName));
                const probabilities = this._network.update(pixels);
                // Step 2: loss.
                const loss = crossEntropyLoss(probabilities, image.digit);
                batchLossTotal += loss;
                epochLossTotal += loss;
                // Step 3: backpropagate the errors from the desired output.
                this._network.backpropagate(oneHot(image.digit, Network.NUM_CLASSES));
                if (DEBUG.OUTPUT_ERROR_SIGNAL) {
                    const errors = this._network.outputErrors;
                    console.log(`Error signal (digit ${image.digit}): [${errors.map(e => e.toFixed(4)).join(", ")}]`);
                }
                // Step 4: gradients. The first image of a batch replaces the
                // previous batch's gradients; the rest are averaged in.
                const isStartOfBatch = batchCounter === 0;
                this._network.calculateGradients(!isStartOfBatch);
                batchCounter++;
                // End of a batch: either it is full, or this is the last image.
                if (batchCounter === batchSize || image === this._train[this._train.length - 1]) {
                    // Step 5: update the weights and biases using the gradients
                    // averaged over this batch.
                    this._network.updateWeights(this._optimizer);
                    batchNumber++;
                    if (DEBUG.BATCH_LOSS) {
                        const batchLoss = batchLossTotal / batchCounter;
                        console.log(`Epoch ${epoch}/${epochs}, batch ${batchNumber}/${numBatches}: loss ${batchLoss.toFixed(4)}`);
                    }
                    batchCounter = 0;
                    batchLossTotal = 0;
                }
            }
            const epochLoss = epochLossTotal / this._train.length;
            console.log(`Epoch ${epoch}/${epochs}: average loss ${epochLoss.toFixed(4)}`);
            // Check how well the network does on images it has not trained on.
            if (this._validation.length > 0) {
                const validationLoss = this.calculateValidationLoss();
                console.log(`Epoch ${epoch}/${epochs}: validation loss ${validationLoss.toFixed(4)}`);
            }
        }
    }
    /**
     * Calculates the network's average cross-entropy loss on the validation
     * images. Each image is fed forward through the network and compared with
     * its digit, exactly as in training, but there is no backpropagation and
     * no weight update: the network does not learn from these images.
     *
     * Comparing this with the training loss shows whether the network is
     * learning to recognise digits in general or just memorising the training
     * images. If the training loss keeps falling while the validation loss
     * starts to rise, the network has begun to overfit.
     *
     * @returns The average loss over the validation images.
     * @throws Error if there are no validation images (loadData() not called,
     * or dataSplit.validation is 0).
     */
    calculateValidationLoss() {
        if (this._validation.length === 0) {
            throw new Error("No validation images; call loadData() first, with dataSplit.validation above 0");
        }
        let lossTotal = 0;
        for (const image of this._validation) {
            const pixels = this._imageLoader.flatten(this._imageLoader.load(image.fileName));
            const probabilities = this._network.update(pixels);
            lossTotal += crossEntropyLoss(probabilities, image.digit);
        }
        return lossTotal / this._validation.length;
    }
    /** The network being trained. */
    get network() {
        return this._network;
    }
    /** Images the network learns from. */
    get trainingSet() {
        return this._train;
    }
    /** Images used to check progress during training. */
    get validationSet() {
        return this._validation;
    }
    /** Images held back to measure final performance. */
    get testSet() {
        return this._test;
    }
}
/**
 * Cross-entropy loss for one example: -log of the probability the network
 * gave to the correct digit. It is 0 when the network is certain and right,
 * and grows without limit as the correct digit's probability approaches 0.
 * A network guessing evenly across 10 digits scores -log(0.1), about 2.30.
 *
 * @param probabilities The network's output, one probability per digit.
 * @param digit The correct digit (the label).
 */
function crossEntropyLoss(probabilities, digit) {
    const probability = probabilities[digit];
    if (probability === undefined) {
        throw new RangeError(`Label ${digit} has no matching output; the network has ${probabilities.length} outputs`);
    }
    // A probability of exactly 0 would give -log(0) = Infinity, so use a tiny minimum.
    return -Math.log(Math.max(probability, 1e-12));
}
/**
 * Creates a one-hot vector: 1 at the position of the given digit and 0
 * everywhere else, e.g. oneHot(3, 10) is [0, 0, 0, 1, 0, 0, 0, 0, 0, 0].
 * This is the network's desired output for an image of that digit.
 *
 * @param digit The correct digit (the label).
 * @param length The number of classes.
 */
function oneHot(digit, length) {
    const vector = new Array(length).fill(0);
    vector[digit] = 1;
    return vector;
}
/**
 * Shuffles an array in place into a random order, using the Fisher-Yates
 * algorithm, which makes every order equally likely.
 */
function shuffle(items) {
    for (let i = items.length - 1; i > 0; i--) {
        const j = Math.floor(random() * (i + 1));
        [items[i], items[j]] = [items[j], items[i]];
    }
}
//# sourceMappingURL=Trainer.js.map