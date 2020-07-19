import {GridworldState, textToTerrain} from "./gridworld-mdp";
import {GridworldGame} from "./gridworld-game";
import {GridworldTrajectoryPlayer} from "./gridworld-trajectory-player";

export class GridworldTrajectoryDisplay extends GridworldTrajectoryPlayer {

    constructor() {
        super();

    }

    static get observedAttributes() {
        return ["terrain", "trajectory", "stepwise"]
    }

    connectedCallback() {
        if (!this.playButton) {
            this.buildSkeleton()
        }
    }

    buildSkeleton() {
        super.buildSkeleton()
        this.playButton.style.display = "none"
    }

    buildGame() {
        // Clean out any previous running game
        this.game?.close()
        let terrainData = this.getAttribute("terrain").split("'")
        terrainData = terrainData.filter((value: any, index: number) => {
            return index % 2 == 1;
        });
        this.trajectory = this.getAttribute("trajectory").split(",").map((value: string) => {
            return parseInt(value)
        })

        let stepwise = this.getAttribute("stepwise") !== "false";
        const terrain = textToTerrain(terrainData)
        this.game = new GridworldGame(terrain["terrain"], this.gameContainer, 16)
        this.game.scene.stepwise = false
        this.game.interactive = false

        this.playButton.style.display = "none";
        this.game.displayTrajectory = this.trajectory

        this.game.sceneCreatedDelegate = () => {
            var image = new Image();
            image.src = this.game.container.getElementsByTagName("canvas")[0].toDataURL("image/png");
            this.game?.close()
            const imageElement = document.createElement("img")
            imageElement.src = image.src
            this.gameContainer.appendChild(imageElement);
        }
        this.game.init();

    }

}

window.customElements.define("gridworld-display", GridworldTrajectoryDisplay)