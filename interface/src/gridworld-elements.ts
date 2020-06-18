import {GridworldState, textToTerrain} from "./gridworld-mdp";
import {GridworldGame} from "./gridworld-game";

export class GridworldTrajectoryPlayer extends HTMLElement {
    private game: GridworldGame;
    private trajectory: number[];
    private currentState: any
    private currentIndex: number
    private intervalId: number;
    private state: GridworldState
    connectedCallback() {
        let terrainData = this.getAttribute("data-terrain").split("'")
        terrainData = terrainData.filter((value: any, index: number) => {
            return index % 2 == 1;
        });
        let container = document.createElement("div")
        this.append(container)
        const terrain = textToTerrain(terrainData)
        this.game = new GridworldGame(terrain["terrain"], container, 64)
        this.game.scene.interactive = false
        this.game.init();
        this.trajectory = this.getAttribute("data-trajectory").split(",").map((value:string) => {return parseInt(value)})
        let playButton = document.createElement("button")
        playButton.innerText = "Play"
        playButton.classList.add("play")
        this.appendChild( playButton)
        playButton.onclick = () => {this.play()}
    }

    play() {
        this.reset()
        setTimeout(() => {
            this.intervalId = setInterval(() => {this.advance()}, 750);
        }, 750)


    }

    advance() {
        const actioni = this.trajectory[this.currentIndex];
        const statei =  this.game.mdp.transition(this.state, actioni)
        this.game.drawState(statei);
        this.state = statei;
        this.currentIndex += 1;
        if (this.currentIndex >= this.trajectory.length) {
            clearInterval(this.intervalId)
            this.intervalId = null
        }
    }

    reset() {
        if (this.intervalId !== null) {
            clearInterval(this.intervalId)
        }
        this.currentIndex = 0;
        this.state = this.game.mdp.getStartState();
        this.game.drawState(this.state, false);
    }


}

window.customElements.define("gridworld-player", GridworldTrajectoryPlayer)