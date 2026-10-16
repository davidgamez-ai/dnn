import { XORNetwork } from "./XORNetwork.js";
/**
 * Creates an XOR network and runs it, to demonstrate how it works.
 */
export class XORTester {
    hyperparameters;
    _network;
    /**
     * Creates the network, which builds it with random starting weights.
     *
     * @param hyperparameters Settings for the network, including the random seed.
     */
    constructor(hyperparameters) {
        this.hyperparameters = hyperparameters;
        this._network = new XORNetwork(hyperparameters);
    }
    /**
     * Feeds the inputs forward through the network once.
     * Set DEBUG.XOR_UPDATE to see the activations, logits and probabilities.
     */
    run() {
        //Demonstrate the network in its untrained state.
        this._network.update([0, 0]);
        this._network.update([0, 1]);
        this._network.update([1, 0]);
        this._network.update([1, 1]);
        //Demonstrate error back propagation
        //XOR should give 0 for the input [0,0]
        this._network.update([0, 0]);
        this.network.backpropagate([1, 0]);
    }
    /** The network being tested. */
    get network() {
        return this._network;
    }
}
//# sourceMappingURL=XORTester.js.map