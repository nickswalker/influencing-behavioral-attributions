import {GridworldState, textToStates, textToTerrain} from "./gridworld-mdp";
import {GridworldGame} from "./gridworld-game";

export class GridworldTrajectoryPlayer extends HTMLElement {
    protected game: GridworldGame;
    protected trajectory: GridworldState[];
    protected currentState: any
    protected currentIndex: number
    protected intervalId: number;
    protected state: GridworldState
    protected shadow: ShadowRoot
    protected gameContainer: HTMLElement
    protected playButton: HTMLElement

    constructor() {
        super();
        this.shadow = this.attachShadow({mode: 'open'});

    }

    attributeChangedCallback(name: string, oldValue: string, newValue: string) {
        if (!this.playButton) {
            this.buildSkeleton()
        }
        if (!this.getAttribute("terrain") || !this.getAttribute("trajectory")) {
            return;
        }
        if (!this.game) {
            this.buildGame()
        }

    }

    static get observedAttributes() {
        return ["terrain", "trajectory"]
    }

    connectedCallback() {
        if (!this.playButton) {
            this.buildSkeleton()
        }
    }

    buildSkeleton() {
        this.style.display = "block"
        let playButton = document.createElement("button")
        playButton.innerText = "Play"
        playButton.classList.add("play")
        this.shadow.appendChild(playButton)
        playButton.onclick = () => {
            this.play()
        }
        this.playButton = playButton
        this.gameContainer = document.createElement("div")
        this.gameContainer.style.width = "100%"
        this.gameContainer.style.height = "100%"
        //this.gameContainer.style.backgroundColor = "grey"
        this.shadow.appendChild(this.gameContainer)
    }

    buildGame() {
        // Clean out any previous running game
        this.game?.close()
        let terrainData = this.getAttribute("terrain").split("'")
        terrainData = terrainData.filter((value: any, index: number) => {
            return index % 2 == 1;
        });
        this.trajectory = textToStates(this.getAttribute("trajectory"))

        this.game = new GridworldGame( this.gameContainer, 32)
        this.game.interactive = false
        this.game.init();
    }

    play() {
        this.reset()
        this.game.scene.scene.start()
        setTimeout(() => {
            this.intervalId = setInterval(() => {
                this.advance()
            }, 450);
        }, 350)
    }

    advance() {
        if (this.currentIndex >= this.trajectory.length) {
            // We do this check up front instead of immediately after the draw call
            // to allow the tween to finish
            clearInterval(this.intervalId)
            this.intervalId = null
            this.game.scene.scene.pause()
            return;
        }
        const statei = this.trajectory[this.currentIndex];
        this.game.drawState(statei);
        this.state = statei;
        this.currentIndex += 1;

    }

    reset() {
        if (this.intervalId !== null) {
            clearInterval(this.intervalId)
        }
        this.currentIndex = 0;
        this.state = this.trajectory[0];
        this.game.drawState(this.state, false);
    }


}

window.customElements.define("gridworld-player", GridworldTrajectoryPlayer)