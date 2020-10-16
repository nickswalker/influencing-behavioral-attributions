import {GridMap, GridworldState} from "./gridworld-mdp";
import {GridworldGame} from "./gridworld-game";
import {GridworldTrajectoryPlayer} from "./gridworld-trajectory-player";
import {textToStates, textToTerrain} from "./utils";

export class GridworldTrajectoryDisplay extends GridworldTrajectoryPlayer {

    constructor() {
        super();

    }

    static get observedAttributes() {
        return ["terrain", "trajectory", "stepwise", "map-name"]
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
        this.gameContainer.innerHTML = ""
        let terrain: GridMap = null
        if (this.getAttribute("terrain")) {
            let terrainData = this.getAttribute("terrain").split("'")
            terrainData = terrainData.filter((value: any, index: number) => {
                return index % 2 == 1;
            });
            terrain = textToTerrain(terrainData)
        }
        this.trajectory = textToStates(this.getAttribute("trajectory"))

        let stepwise = this.getAttribute("stepwise") !== "false";
        this.game = new GridworldGame(this.gameContainer, 32, "assets/", this.getAttribute("map-name"),terrain, false)
        this.game.scene.stepwise = false

        this.playButton.style.display = "none";
        this.game.displayTrajectory = this.trajectory

        this.game.sceneCreatedDelegate = () => {
            var image = new Image();
            image.src = this.game.container.getElementsByTagName("canvas")[0].toDataURL("image/png");
            this.game?.close()
            const imageElement = document.createElement("img")
            imageElement.src = image.src
            this.gameContainer.appendChild(imageElement);
            // Clear out any manual styling from Phaser
            this.gameContainer.setAttribute("style", "")
        }
        this.game.init();

    }

}

window.customElements.define("gridworld-display", GridworldTrajectoryDisplay)