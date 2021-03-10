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
        if (!this.gameContainer) {
            this.buildSkeleton()
        }
    }

    buildSkeleton() {
        this.gameContainer = document.createElement("div")
        this.gameContainer.style.width = "100%"
        this.gameContainer.style.height = "100%"
        this.shadow.appendChild(this.gameContainer)
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

        this.game = new GridworldGame(this.gameContainer, null, null, this.getAttribute("map-name"),terrain)
        this.game.scene.stepwise = false
        this.game.displayTrajectory = this.trajectory

        this.game.sceneCreatedDelegate = () => {
            this.game.game.renderer.snapshot((image: HTMLImageElement) =>{
                this.game?.close()
                this.gameContainer.appendChild(image);
                // Clear out any manual styling from Phaser
                this.gameContainer.setAttribute("style", "")
                this.dispatchEvent(new Event("ready"))
            });

        }
        this.game.init();

    }

    download(name: string) {
        const a = document.createElement("a");
        document.body.appendChild(a);
        a.style.display = "none";
        a.href = this.gameContainer.querySelector<HTMLImageElement>("img").src;
        a.download = name + ".png";
        a.click();

    }
}

window.customElements.define("gridworld-display", GridworldTrajectoryDisplay)