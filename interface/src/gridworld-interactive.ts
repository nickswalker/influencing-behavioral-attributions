import {GridMap, GridworldState, TerrainMap} from "./gridworld-mdp";
import {GridworldGame} from "./gridworld-game";
import {textToStates, textToTerrain} from "./utils";


export class GridworldTrajectoryPlayer extends HTMLElement {
    protected game: GridworldGame;
    protected trajectory: GridworldState[];
    protected currentState: any
    protected state: GridworldState
    protected shadow: ShadowRoot
    protected gameContainer: HTMLElement
    protected chunks: Blob[]

    constructor() {
        super();
        this.shadow = this.attachShadow({mode: 'open'});

    }

    attributeChangedCallback(name: string, oldValue: string, newValue: string) {
        if (!this.gameContainer) {
            this.buildSkeleton()
        }
        if (!this.getAttribute("terrain") && !this.getAttribute("map-name")) {
            return;
        }
        if (!this.game) {
            this.buildGame()
        }

    }

    static get observedAttributes() {
        return ["terrain", "map-name"]
    }

    connectedCallback() {
        if (!this.gameContainer) {
            this.buildSkeleton()
        }
    }

    buildSkeleton() {
        this.style.display = "block"
        this.gameContainer = document.createElement("div")
        this.gameContainer.style.width = "100%"
        this.gameContainer.style.height = "100%"
        //this.gameContainer.style.backgroundColor = "grey"
        this.shadow.appendChild(this.gameContainer)
    }

    buildGame() {
        // Clean out any previous running game
        this.game?.close()
        let terrain: GridMap = null
        let mapName = null
        if (this.getAttribute("terrain")) {
            let terrainData = this.getAttribute("terrain").split("'")
            terrainData = terrainData.filter((value: any, index: number) => {
                return index % 2 == 1;
            });
            terrain = textToTerrain(terrainData)
        } else {
            mapName = this.getAttribute("map-name")
        }
        this.game = new GridworldGame( this.gameContainer, 32, "assets/", mapName, terrain)
        this.game.interactive = true
        this.game.init();
        this.game.sceneCreatedDelegate = () => {
            this.game.scene.events.addListener("stateDrawn", () => {
                const startState = this.game.scene.mdp.getStartState();
                const startPos = startState.agentPositions[0]
                const currentTraj = this.getAttribute("trajectory") ?? "[(" + String(startPos.x) + "," + String(startPos.y) + ")]";
                const positions = this.game.scene.currentState.agentPositions[0]
                this.setAttribute("trajectory", currentTraj.slice(0,-1) + ", (" + String(positions.x) + ", " + String(positions.y) + ")]")
            })
        }
    }


    reset() {
        this.state = this.trajectory[0];
        this.game.drawState(this.state, false);
        this.trajectory = [];
    }


}

window.customElements.define("gridworld-interactive", GridworldTrajectoryPlayer)