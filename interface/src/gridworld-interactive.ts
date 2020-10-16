import {GridMap, GridworldState, TerrainMap} from "./gridworld-mdp";
import {GridworldGame} from "./gridworld-game";
import {statesToText, textToStates, textToTerrain} from "./utils";


export class GridworldInteractive extends HTMLElement {
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
        //this.gameContainer.style.width = "100%"
        //this.gameContainer.style.height = "100%"
        //this.gameContainer.style.backgroundColor = "grey"
        this.shadow.appendChild(this.gameContainer)
        const resetButton = document.createElement("button")
        resetButton.value = "reset"
        resetButton.innerText = "Reset"
        resetButton.addEventListener("click", () => {
            this.reset()
        })
        this.shadow.appendChild(resetButton)

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
        this.game = new GridworldGame(this.gameContainer, 32, "assets/", mapName, terrain)
        this.trajectory = []
        this.game.init();
        this.game.sceneCreatedDelegate = () => {
            this.game.scene.clearTrail()
            this.game.scene.events.addListener("stateDrawn", () => {
                this.setAttribute("trajectory", statesToText(this.trajectory))
                this.trajectory.push(this.game.scene.currentState)
            })
            this.disable()
            this.game.scene.scene.resume()
            this.game.game.events.addListener(Phaser.Core.Events.FOCUS, () => {

            })
            this.game.game.events.addListener(Phaser.Core.Events.BLUR, () => {

            })
        }
        this.addEventListener("click", (event) => {
            event.stopPropagation();
            this.enable()
        })
    }


    reset() {
        this.state = this.game.scene.mdp.getStartState();
        this.trajectory = [];
        this.removeAttribute("trajectory")
        this.game.state = this.state
        this.game.scene.currentState = this.state
        this.game.drawState(this.state, true);
        this.game.scene.clearTrail()

    }

    disable() {
        this.game.scene.interactive = false;
        this.game.game.input.enabled = false;
        this.game.game.input.keyboard.preventDefault = false;
        this.game.scene.scene.pause()
        this.game.game.loop.sleep()
        this.gameContainer.style.opacity = "0.7"

    }

    enable() {
        this.game.scene.interactive = true;
        this.game.game.input.enabled = true;
        this.game.game.input.keyboard.preventDefault = true;
        this.game.scene.scene.resume()
        this.game.game.loop.wake()
        this.gameContainer.style.opacity = "1.0"
    }

}

window.addEventListener("click", (event: MouseEvent) => {
    document.querySelectorAll<GridworldInteractive>("gridworld-interactive").forEach((element) => {
        element.disable()
    })
})

window.customElements.define("gridworld-interactive", GridworldInteractive)