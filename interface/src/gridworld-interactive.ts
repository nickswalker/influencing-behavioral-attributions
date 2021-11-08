import {GridMap, GridworldState, TerrainMap} from "./gridworld-mdp";
import {GridworldGame} from "./gridworld-game";
import {statesToText, textToStates, textToTerrain} from "./utils";
import {GridworldTrajectoryPlayer} from "./gridworld-trajectory-player";


export class GridworldInteractive extends HTMLElement {
    protected game: GridworldGame;
    protected trajectory: GridworldState[];
    protected currentState: any
    protected state: GridworldState
    protected shadow: ShadowRoot
    protected gameContainer: HTMLElement
    protected chunks: Blob[]
    protected enabled: boolean
    protected playbackElement: any
    protected playButton: HTMLButtonElement
    constructor() {
        super();
        this.shadow = this.attachShadow({mode: 'open'});

    }

    get playbackVisible(): boolean {
        return this.playbackElement.style.display !== "none"
    }

    attributeChangedCallback(name: string, oldValue: string, newValue: string) {
        if (!this.getAttribute("terrain") && !this.getAttribute("map-name") && !this.getAttribute("start-x") && !this.getAttribute("start-y")) {
            return;
        }
        if (!this.gameContainer) {
            this.buildSkeleton()
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

        const playback: any = document.createElement("gridworld-player")
        playback.setAttribute("map-name", this.getAttribute("map-name"))
        playback.setAttribute("trajectory", "[("+ this.getAttribute("start-x")+", " + this.getAttribute("start-y")+")]")

        this.shadow.append(playback)

        playback.style.display = "none"
        this.playbackElement = playback

        const resetButton = document.createElement("button")
        resetButton.value = "reset"
        resetButton.innerText = "Reset"
        resetButton.addEventListener("click", () => {
            this.reset()
        })
        this.shadow.appendChild(resetButton)

        const playbackButton = document.createElement("button")
        playbackButton.value = "play"
        playbackButton.innerText = "Play"
        playbackButton.addEventListener("click", () => {
            this.playback()
        })
        playbackButton.disabled = true
        this.playButton = playbackButton
        this.shadow.appendChild(playbackButton)


    }

    buildGame() {
        // Clean out any previous running game
        this.game?.close()
        this.enabled = true
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
        this.trajectory = []
        let initialState = new GridworldState([{x:Number(this.getAttribute("start-x")), y:Number(this.getAttribute("start-y"))}])
        this.game = new GridworldGame(this.gameContainer, null, null, mapName, terrain, initialState)
        this.game.sceneCreatedDelegate = () => {
            this.game.scene.interactive = true
            this.game.scene.reset()
            this.trajectory = [this.game.scene.currentState]
            this.game.scene.events.addListener("stateDrawn", () => {
                this.trajectory.push(this.game.scene.currentState)
                this.setAttribute("trajectory", statesToText(this.trajectory))
                this.game.scene.events.once("render", ()=> {
                    if (this.game.scene.interactionFinished) {
                        this.classList.add("finished")
                        this.playButton.disabled = false
                    } else {
                        this.classList.remove("finished")
                        this.playButton.disabled = true
                    }
                })
                this.dispatchEvent(new Event("trajectoryChange",{bubbles:true}))

            })
            this.disable()
        }
        this.game.init();

        this.gameContainer.addEventListener("click", (event) => {
            document.querySelectorAll<GridworldInteractive>("gridworld-interactive").forEach((element) => {
                if (element === this) {
                    return;
                }
                element.disable()
            })
            event.stopPropagation();
            this.enable()
        })
    }


    reset() {
        this.playbackElement.style.display = "none"
        this.gameContainer.querySelector("canvas").hidden = false
        this.trajectory = [];
        this.removeAttribute("trajectory")
        this.game.scene.reset()
        this.enable()

    }

    disable(done: () => void = null) {
        if (!this.enabled) {
            return
        }
        this.enabled = false

        //this.game.game.loop.sleep()

        this.game.game.renderer.snapshot((image: HTMLImageElement) => {
            this.game.scene.scene.sleep()
            this.game.scene.interactive = false;

            this.gameContainer.querySelector("canvas").hidden = true
            image.style.opacity = "0.7"
            this.gameContainer.appendChild(image);
            // Clear out any manual styling from Phaser
            this.gameContainer.setAttribute("style", "")
            if (done) {
                done()
            }
        });

    }

    enable() {
        if (this.enabled) {
            return
        }
        this.enabled = true
        this.game.scene.interactive = true;

        this.game.scene.scene.wake()
        this.gameContainer.querySelectorAll("img").forEach((element) => {
            element.remove()
        })
        this.gameContainer.querySelector("canvas").hidden = false
    }

    playback() {
        if (!this.game.scene.interactionFinished) {
            return
        }
        this.playbackElement.playButton.hidden = true
        const playbackClickHandler = () => {
            this.playbackElement.play();
            this.playbackElement.removeEventListener("ready", playbackClickHandler)
        }
        if (this.enabled) {
            this.disable(() => {
                this.gameContainer.querySelectorAll("img").forEach((element) => {
                    element.remove()
                })
                this.playbackElement.style.display = ""
                this.playbackElement.addEventListener("ready", playbackClickHandler)
                this.playbackElement.setAttribute("trajectory", this.getAttribute("trajectory"))

            })
        }
        else {
            if (!this.playbackVisible) {
                this.gameContainer.querySelectorAll("img").forEach((element) => {
                    element.remove()
                })
                this.playbackElement.style.display = ""
                this.playbackElement.addEventListener("ready", playbackClickHandler)
                this.playbackElement.setAttribute("trajectory", this.getAttribute("trajectory"))
            } else {
                this.playbackElement.reset(()=>(this.playbackElement.play()))
            }

        }


    }

}

document.addEventListener("click", (event: MouseEvent) => {
    document.querySelectorAll<GridworldInteractive>("gridworld-interactive").forEach((element) => {
        if (event.target === element) {
            return;
        }
        element.disable()
    })
})

window.customElements.define("gridworld-interactive", GridworldInteractive)