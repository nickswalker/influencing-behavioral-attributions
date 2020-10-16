import {GridMap, GridworldState} from "./gridworld-mdp";
import {GridworldGame} from "./gridworld-game";
import {hashCode, textToStates, textToTerrain} from "./utils";

declare global {
    interface HTMLCanvasElement {
        captureStream(frameRate?: number): MediaStream;
    }
}

declare interface MediaRecorderErrorEvent extends Event {
    name: string;
}

declare interface MediaRecorderDataAvailableEvent extends Event {
    data : any;
}

interface MediaRecorderEventMap {
    'dataavailable': MediaRecorderDataAvailableEvent;
    'error': MediaRecorderErrorEvent ;
    'pause': Event;
    'resume': Event;
    'start': Event;
    'stop': Event;
    'warning': MediaRecorderErrorEvent ;
}


declare class MediaRecorder extends EventTarget {

    readonly mimeType: string;
    readonly state: 'inactive' | 'recording' | 'paused';
    readonly stream: MediaStream;
    ignoreMutedMedia: boolean;
    videoBitsPerSecond: number;
    audioBitsPerSecond: number;

    ondataavailable: (event : MediaRecorderDataAvailableEvent) => void;
    onerror: (event: MediaRecorderErrorEvent) => void;
    onpause: () => void;
    onresume: () => void;
    onstart: () => void;
    onstop: () => void;

    constructor(stream: MediaStream, options: any);

    start(): null;

    stop(): null;

    resume(): null;

    pause(): null;

    isTypeSupported(type: string): boolean;

    requestData(): null;


    addEventListener<K extends keyof MediaRecorderEventMap>(type: K, listener: (this: MediaStream, ev: MediaRecorderEventMap[K]) => any, options?: boolean | AddEventListenerOptions): void;

    addEventListener(type: string, listener: EventListenerOrEventListenerObject, options?: boolean | AddEventListenerOptions): void;

    removeEventListener<K extends keyof MediaRecorderEventMap>(type: K, listener: (this: MediaStream, ev: MediaRecorderEventMap[K]) => any, options?: boolean | EventListenerOptions): void;

    removeEventListener(type: string, listener: EventListenerOrEventListenerObject, options?: boolean | EventListenerOptions): void;

}

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
    protected recorder: MediaRecorder
    protected chunks: Blob[]
    protected trajHash: number

    constructor() {
        super();
        this.shadow = this.attachShadow({mode: 'open'});


    }

    attributeChangedCallback(name: string, oldValue: string, newValue: string) {
        if (!this.playButton) {
            this.buildSkeleton()
        }
        if (!this.getAttribute("terrain") && !this.getAttribute("map-name")) {
            return;
        }
        if ( !this.getAttribute("trajectory")) {
            return;
        }
        if (!this.game || hashCode(this.getAttribute("trajectory")) !== this.trajHash) {
            this.buildGame()
        }

    }

    static get observedAttributes() {
        return ["terrain", "trajectory", "map-name"]
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
        this.trajHash = hashCode(this.getAttribute("trajectory"))
        let terrain: GridMap = null
        if (this.getAttribute("terrain")) {
            let terrainData = this.getAttribute("terrain").split("'")
            terrainData = terrainData.filter((value: any, index: number) => {
                return index % 2 == 1;
            });
            terrain = textToTerrain(terrainData)
        }
        this.trajectory = textToStates(this.getAttribute("trajectory"))
        this.orientation = [0, -1]
        this.diffs = [new GridworldState([{x: 1, y:0}])]
        for (let i = 1; i < this.trajectory.length; i++) {
            const prev = this.trajectory[i - 1].agentPositions[0]
            const current = this.trajectory[i].agentPositions[0]
            this.diffs.push(new GridworldState([{x: current.x - prev.x, y: current.y - prev.y}]))
        }

        this.game = new GridworldGame( this.gameContainer, 32, "assets/", this.getAttribute("map-name"),terrain)
        this.game.interactive = false
        this.game.init();
        this.game.game.events.once("postrender", () =>{
            this.game.scene.scene.pause()
        })
    }

    configureRecorder() {
        const options = {mimeType: "video/webm; codecs=vp9"};

        let stream = this.shadow.querySelector("canvas").captureStream(25);
        this.recorder = new MediaRecorder(stream, options);
        this.recorder.addEventListener("dataavailable", (event: MediaRecorderDataAvailableEvent) => {
            if (event.data.size > 0) {
                this.chunks.push(event.data);
            } else {

            }
        });
        this.chunks = [];

    }

    play() {
        this.reset()
        this.game.scene.scene.resume()

        this.recorder?.start();

        setTimeout(() => {
            this.intervalId = setInterval(() => {
                this.advance()
            }, 300);
        }, 200)

    }

    download(name:string) {
        const blob = new Blob(this.chunks, {
            type: "video/webm"
        });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        document.body.appendChild(a);
        a.style.display = "none";
        a.href = url;
        a.download = name + ".webm";
        a.click();
        window.URL.revokeObjectURL(url);
    }

    advance() {
        if (this.currentIndex >= this.trajectory.length) {
            // We do this check up front instead of immediately after the draw call
            // to allow the tween to finish
            clearInterval(this.intervalId)
            this.intervalId = null
            this.game.scene.scene.pause()
            this.recorder?.stop()
            return;
        }
        const diff = this.diffs[this.currentIndex]
        if (diff.agentPositions[0].x == 0 && diff.agentPositions[0].y == 0) {

        }
        else if (this.orientation[0] != diff.agentPositions[0].x || this.orientation[1] != diff.agentPositions[0].y) {
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
        this.game.scene.clearTrail()
    }


}

window.customElements.define("gridworld-player", GridworldTrajectoryPlayer)