import DEBUG from "./Debug.js";
import { Hyperparameters } from "./Hyperparameters.js";
import { XORNetwork } from "./XORNetwork.js";
import { XORTrainer } from "./XORTrainer.js";
/**
 * Creates an XOR network and runs it, to demonstrate how it works.
 */
export class XORTester {
    hyperparameters;
    _network;
    _trainer;
    /**
     * Hyperparameters that train the XOR network reliably:
     *   learningRate 0.01 - the default of 0.001 suits the larger digit
     *                       network, but is slow for XOR.
     *   epochs 2000       - each epoch is a single weight update, so XOR
     *                       needs many of them.
     *   randomSeed 9      - with only two ReLU hidden neurons many starting
     *                       weights get stuck (see XORNetwork); this one learns.
     * Returns a new object each time, so changing it does not affect later calls.
     */
    static defaultHyperparameters() {
        return new Hyperparameters({ learningRate: 0.01, epochs: 2000, randomSeed: 9 });
    }
    /**
     * Creates the network, which builds it with random starting weights, and
     * the trainer that will train it.
     *
     * @param hyperparameters Settings for the network and training, including
     * the random seed. Defaults to defaultHyperparameters().
     */
    constructor(hyperparameters = XORTester.defaultHyperparameters()) {
        this.hyperparameters = hyperparameters;
        this._network = new XORNetwork(hyperparameters);
        this._trainer = new XORTrainer(hyperparameters, this._network);
    }
    /**
     * Shows the network's results before training, trains it, then shows the
     * results again: the output for each of the four examples, how many are
     * correct, and the weights and biases.
     *
     * The per-example debug logging (DEBUG.XOR_UPDATE, XOR_ERROR_SIGNAL and
     * XOR_GRADIENTS) is switched off while this runs, as training would print
     * thousands of blocks. It is switched back to its previous setting
     * afterwards, ready for demo().
     *
     * @param logInterval How many epochs between loss reports during training.
     */
    run(logInterval = 200) {
        const { XOR_UPDATE, XOR_ERROR_SIGNAL, XOR_GRADIENTS } = DEBUG;
        DEBUG.XOR_UPDATE = false;
        DEBUG.XOR_ERROR_SIGNAL = false;
        DEBUG.XOR_GRADIENTS = false;
        try {
            console.log("Before training:");
            this._network.printResults();
            console.log(`\nTraining for ${this.hyperparameters.epochs} epochs:`);
            this._trainer.train(logInterval);
            console.log("\nAfter training:");
            this._network.printResults();
        }
        finally {
            DEBUG.XOR_UPDATE = XOR_UPDATE;
            DEBUG.XOR_ERROR_SIGNAL = XOR_ERROR_SIGNAL;
            DEBUG.XOR_GRADIENTS = XOR_GRADIENTS;
        }
    }
    /**
     * Feeds the inputs forward through the network once.
     * Set DEBUG.XOR_UPDATE to see the activations, logits and probabilities.
     */
    demo() {
        //Demonstrate the network in its untrained state.
        this._network.update([0, 0]);
        this._network.update([0, 1]);
        this._network.update([1, 0]);
        this._network.update([1, 1]);
        //Demonstrate error back propagation
        //Pass 0, 0 into the network
        this._network.update([1, 1]);
        //XOR should give 0 for the input [0,0]
        //Output 0 encodes probability of 0 as the output
        //Output 1 encodes probability of 1 as teh output
        //So correct answer from network is 1,0.
        this.network.backpropagate([1, 0]);
        //Gradient calculation
        this.network.calculateGradients(false);
    }
    /** The network being tested. */
    get network() {
        return this._network;
    }
}
//# sourceMappingURL=XORTester.js.map