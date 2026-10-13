import { readdirSync } from "node:fs";
import { join } from "node:path";
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
    _imageLoader = new ImageLoader();
    _train = [];
    _validation = [];
    _test = [];
    /**
     * Creates the network to be trained and builds it, so its weights and
     * biases are ready to use.
     *
     * @param hyperparameters Settings for training, including how to split the data and the random seed.
     */
    constructor(hyperparameters) {
        this.hyperparameters = hyperparameters;
        this._hyperparameters = hyperparameters;
        this._network = new Network(hyperparameters);
        this._network.build();
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
     * Works through the training images once, in batches of batchSize (from
     * the hyperparameters). The last batch may be smaller if the images do not
     * divide evenly.
     *
     * For each image:
     *   1. Load and flatten the image, and feed it forward through the network.
     *   2. Measure the cross-entropy loss against the image's digit.
     *   3. Backpropagate, so every layer has its errors for this image.
     *   4. Calculate the gradients, averaging them over the images in the batch.
     *
     * At the end of each batch the average loss is logged. This is where the
     * weights will be updated, using the batch's averaged gradients.
     *
     * The weights are not updated yet, so the loss stays at roughly the same level.
     *
     * @throws Error if loadData() has not been called, or found no training images.
     */
    train() {
        if (this._train.length === 0) {
            throw new Error("No training images; call loadData() first");
        }
        const batchSize = this._hyperparameters.batchSize;
        const numBatches = Math.ceil(this._train.length / batchSize);
        let batchCounter = 0;
        let batchLossTotal = 0;
        let batchNumber = 0;
        for (const image of this._train) {
            // Step 1: forward pass.
            const pixels = this._imageLoader.flatten(this._imageLoader.load(image.fileName));
            const probabilities = this._network.update(pixels);
            // Step 2: loss.
            batchLossTotal += crossEntropyLoss(probabilities, image.digit);
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
                // TODO: update the weights here, using each layer's averaged
                // gradients and the learning rate.
                batchNumber++;
                const batchLoss = batchLossTotal / batchCounter;
                console.log(`Batch ${batchNumber}/${numBatches}: loss ${batchLoss.toFixed(4)}`);
                batchCounter = 0;
                batchLossTotal = 0;
            }
        }
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