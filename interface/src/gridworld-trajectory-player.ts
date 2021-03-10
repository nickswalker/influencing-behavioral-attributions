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
    protected currentIndex: number
    protected intervalId: number;
    protected shadow: ShadowRoot
    protected gameContainer: HTMLElement
    protected playButton: HTMLElement
    protected orientation: number[]
    protected diffs: GridworldState[]
    protected recorder: MediaRecorder
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
        if ( !this.getAttribute("trajectory")) {
            return;
        }

        this.buildGame()


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
        let playButton = document.createElement("button")
        playButton.innerText = "Play"
        playButton.classList.add("play")
        this.shadow.appendChild(playButton)
        playButton.onclick = () => {this.playClickHandler()}
        this.playButton = playButton
        this.gameContainer = document.createElement("div")
        this.gameContainer.style.width = "100%"
        this.gameContainer.style.height = "100%"
        this.shadow.appendChild(this.gameContainer)

        this.addEventListener("click", (event)=>{
            event.stopPropagation()
            this.playClickHandler()
        })
    }

    buildGame() {
        // Clean out any previous running game
        this.game?.close()
        this.currentIndex = 0;
        if (this.intervalId !== null) {
            clearInterval(this.intervalId)
            this.intervalId = null
        }
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

        this.game = new GridworldGame( this.gameContainer, null, null, this.getAttribute("map-name"),terrain)
        this.game.init();
        this.game.sceneCreatedDelegate = () => {
            this.freezeFrame(() => {
                this.dispatchEvent(new Event("ready"))
            })

        }

    }

    configureRecorder() {
        // 5000kbps
        // Chrome ignores bitrate, but Firefox doesn't. Firefox won't encode VP9 though.
        // So to get good quality, high bitrate vp8 -> ffmpeg compress to VP9
        const options = {mimeType: "video/webm; codecs=vp8",videoBitsPerSecond: 5000000};

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

    playClickHandler() {
        if (this.intervalId && this.intervalId !== -1) {
            this.reset()
        } else {
            this.currentIndex = 0
            this.play()
        }
    }

    play() {
        if (this.intervalId) {
            return;
        }
        this.intervalId = -1
        this.gameContainer.querySelectorAll("img").forEach((element) => {
            element.remove()
        })
        this.gameContainer.querySelector("canvas").hidden = false
        this.game.scene.scene.wake()
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
            this.recorder?.stop()
            this.freezeFrame()
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
        this.currentIndex += 1;

    }

    reset(done: ()=>void = null) {
        if (this.intervalId !== null) {
            clearInterval(this.intervalId)
            this.intervalId = null
        }
        this.currentIndex = 0;
        this.game.drawState(this.trajectory[0], false);
        this.game.scene.reset()
        this.freezeFrame(done)
    }

    freezeFrame(done: ()=>void = null) {
        this.game.game.renderer.snapshot((image: HTMLImageElement) => {
            this.game.scene.scene.sleep()
            this.gameContainer.querySelectorAll("img").forEach((element) => {
                element.remove()
            })
            this.gameContainer.querySelector("canvas").hidden = true
            image.style.opacity = "0.7"
            this.gameContainer.appendChild(image);
            // Clear out any manual styling from Phaser
            this.gameContainer.setAttribute("style", "")
            if (done) {done()}
        });
    }


}

window.customElements.define("gridworld-player", GridworldTrajectoryPlayer)