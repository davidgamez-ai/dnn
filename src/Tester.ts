import { Hyperparameters } from "./Hyperparameters.js";
import { ImageLoader } from "./ImageLoader.js";
import { Network } from "./Network.js";
import { Trainer } from "./Trainer.js";

/**
 * A 2x2 contingency table for one digit, treating the problem as
 * "this digit" versus "any other digit" (one-vs-rest).
 */
export interface ContingencyTable {
    /** Images of this digit, predicted as this digit. */
    readonly truePositives: number;
    /** Images of another digit, wrongly predicted as this digit. */
    readonly falsePositives: number;
    /** Images of this digit, wrongly predicted as another digit. */
    readonly falseNegatives: number;
    /** Images of another digit, predicted as some other digit (not this one). */
    readonly trueNegatives: number;
}

/**
 * Creates a network, trains it, then measures how well it recognises the
 * test images: images it saw neither during training nor during validation.
 * The results are shown as a confusion matrix.
 */
export class Tester {
    private readonly _network: Network;
    private readonly _trainer: Trainer;
    private readonly _imageLoader = new ImageLoader();
    private _confusionMatrix: number[][] | undefined;

    /**
     * Creates the network and the trainer that will train it.
     *
     * @param hyperparameters Settings for the network, training and data split.
     */
    constructor(readonly hyperparameters: Hyperparameters) {
        this._network = new Network(hyperparameters);
        this._trainer = new Trainer(hyperparameters, this._network);
    }

    /**
     * Loads the data, trains the network, tests it on the test images and
     * logs the confusion matrix, accuracy and a contingency table for each digit.
     */
    run(): void {
        this._trainer.loadData();
        this._trainer.train();
        this.test();
        this.printResults();
        this.printContingencyTables();
    }

    /**
     * Feeds every test image through the trained network and records the
     * results in a confusion matrix.
     *
     * The network's prediction for an image is the digit with the highest
     * probability. confusionMatrix[actual][predicted] counts how many images
     * of digit `actual` the network predicted as digit `predicted`. Correct
     * predictions lie on the diagonal (actual = predicted); everything else is
     * a mistake, and shows which digits the network confuses with which.
     *
     * Like validation, testing only feeds images forward: the network does
     * not learn from them.
     *
     * @returns The confusion matrix, indexed [actual digit][predicted digit].
     * @throws Error if there are no test images (loadData() not called, or dataSplit.test is 0).
     */
    test(): number[][] {
        const testSet = this._trainer.testSet;
        if (testSet.length === 0) {
            throw new Error("No test images; call loadData() first, with dataSplit.test above 0");
        }

        // Start with a NUM_CLASSES x NUM_CLASSES matrix of zeros.
        const matrix: number[][] = [];
        for (let actual = 0; actual < Network.NUM_CLASSES; actual++) {
            matrix.push(new Array<number>(Network.NUM_CLASSES).fill(0));
        }

        for (const image of testSet) {
            const pixels = this._imageLoader.flatten(this._imageLoader.load(image.fileName));
            const probabilities = this._network.update(pixels);
            const predicted = indexOfLargest(probabilities);
            matrix[image.digit]![predicted]!++;
        }

        this._confusionMatrix = matrix;
        return matrix;
    }

    /**
     * Logs the confusion matrix as a table (rows are the actual digits,
     * columns the predicted digits), followed by the overall accuracy.
     *
     * @throws Error if test() has not been called yet.
     */
    printResults(): void {
        const matrix = this.confusionMatrix;

        // console.table uses the outer keys as row labels and the inner keys
        // as column labels.
        const table: Record<string, Record<string, number>> = {};
        let correct = 0;
        let total = 0;
        for (let actual = 0; actual < matrix.length; actual++) {
            const row: Record<string, number> = {};
            for (let predicted = 0; predicted < matrix[actual]!.length; predicted++) {
                const count = matrix[actual]![predicted]!;
                row[`predicted ${predicted}`] = count;
                total += count;
                if (actual === predicted) {
                    correct += count;
                }
            }
            table[`actual ${actual}`] = row;
        }

        console.log("Confusion matrix (rows: actual digit, columns: predicted digit)");
        console.table(table);
        console.log(`Test accuracy: ${correct}/${total} = ${(100 * correct / total).toFixed(1)}%`);
    }

    /**
     * Builds the 2x2 contingency table for one digit from the confusion
     * matrix, by asking of every test image: "is it this digit?" and "did the
     * network say it was this digit?"
     *
     *                          predicted digit    predicted not digit
     *     actual digit         true positive      false negative
     *     actual not digit     false positive     true negative
     *
     * In the confusion matrix, the true positives are the diagonal entry for
     * the digit, the false negatives are the rest of its row, the false
     * positives are the rest of its column, and the true negatives are
     * everything else.
     *
     * @param digit The digit to build the table for (0-9).
     * @throws RangeError if digit is not a valid class.
     * @throws Error if test() has not been called yet.
     */
    contingencyTable(digit: number): ContingencyTable {
        const matrix = this.confusionMatrix;
        if (!Number.isInteger(digit) || digit < 0 || digit >= matrix.length) {
            throw new RangeError(`digit must be an integer from 0 to ${matrix.length - 1}, got ${digit}`);
        }

        let truePositives = 0;
        let falsePositives = 0;
        let falseNegatives = 0;
        let trueNegatives = 0;
        for (let actual = 0; actual < matrix.length; actual++) {
            for (let predicted = 0; predicted < matrix.length; predicted++) {
                const count = matrix[actual]![predicted]!;
                const isActuallyDigit = actual === digit;
                const isPredictedDigit = predicted === digit;
                if (isActuallyDigit && isPredictedDigit) {
                    truePositives += count;
                } else if (isActuallyDigit) {
                    falseNegatives += count;
                } else if (isPredictedDigit) {
                    falsePositives += count;
                } else {
                    trueNegatives += count;
                }
            }
        }
        return { truePositives, falsePositives, falseNegatives, trueNegatives };
    }

    /**
     * Logs a 2x2 contingency table for each digit, with its precision and recall:
     *
     *     precision = true positives / (true positives + false positives)
     *                 "when the network says 3, how often is it really a 3?"
     *     recall    = true positives / (true positives + false negatives)
     *                 "of all the real 3s, how many did the network find?"
     *
     * @throws Error if test() has not been called yet.
     */
    printContingencyTables(): void {
        for (let digit = 0; digit < Network.NUM_CLASSES; digit++) {
            const { truePositives, falsePositives, falseNegatives, trueNegatives } = this.contingencyTable(digit);

            console.log(`Contingency table for digit ${digit}`);
            console.table({
                [`actual ${digit}`]: { [`predicted ${digit}`]: truePositives, [`predicted not ${digit}`]: falseNegatives },
                [`actual not ${digit}`]: { [`predicted ${digit}`]: falsePositives, [`predicted not ${digit}`]: trueNegatives },
            });
            const precision = formatRatio(truePositives, truePositives + falsePositives);
            const recall = formatRatio(truePositives, truePositives + falseNegatives);
            console.log(`Precision: ${precision}   Recall: ${recall}`);
        }
    }

    /**
     * The confusion matrix from the most recent test() call, indexed
     * [actual digit][predicted digit].
     * @throws Error if test() has not been called yet.
     */
    get confusionMatrix(): readonly (readonly number[])[] {
        if (this._confusionMatrix === undefined) {
            throw new Error("No confusion matrix yet; call test() first");
        }
        return this._confusionMatrix;
    }

    /** The network being trained and tested. */
    get network(): Network {
        return this._network;
    }
}

/**
 * Formats a ratio as a percentage, or "n/a" if the denominator is 0 (for
 * example, precision for a digit the network never predicted).
 */
function formatRatio(numerator: number, denominator: number): string {
    return denominator === 0 ? "n/a" : `${(100 * numerator / denominator).toFixed(1)}%`;
}

/** Returns the position of the largest value in an array. */
function indexOfLargest(values: readonly number[]): number {
    let bestIndex = 0;
    for (let i = 1; i < values.length; i++) {
        if (values[i]! > values[bestIndex]!) {
            bestIndex = i;
        }
    }
    return bestIndex;
}
