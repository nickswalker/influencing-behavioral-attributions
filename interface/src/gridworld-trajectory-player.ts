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
    protected orientation: number[]
    protected diffs: GridworldState[]

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
        this.orientation = [0, -1]
        this.diffs = [new GridworldState([{x: 1, y:0}])]
        for (let i = 1; i < this.trajectory.length; i++) {
            const prev = this.trajectory[i - 1].agentPositions[0]
            const current = this.trajectory[i].agentPositions[0]
            this.diffs.push(new GridworldState([{x: current.x - prev.x, y: current.y - prev.y}]))
        }

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
            }, 300);
        }, 200)
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
        const diff = this.diffs[this.currentIndex]
        if (this.orientation[0] != diff.agentPositions[0].x || this.orientation[1] != diff.agentPositions[0].y) {
            this.game.rotateAgent(diff.agentPositions[0].x, diff.agentPositions[0].y)
            this.orientation = [diff.agentPositions[0].x, diff.agentPositions[0].y]
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