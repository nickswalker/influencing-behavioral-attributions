import "./gridworld-mdp"
import "./gridworld-game"
import "./gridworld-trajectory-player"
import "./gridworld-trajectory-display"
import "./paged-navigation"
import {Direction, TerrainMap, textToTerrain} from "./gridworld-mdp";
import {GridworldGame} from "./gridworld-game";
import * as io from "socket.io-client"

let game = document.getElementById("player")

function f() {
    const socket = io("http://127.0.0.1:5000");
    socket.on('connect', function () {
        console.log("sent event")
        socket.emit('my event', {data: 'I\'m connected!'});
    });

    /* Modes
        * Client-only: Maps are set up in source. Human trajectories are encoded and dumped into form element
        * View mode: Remote sends trajectory, client queues and renders them at reasonable rate
        * Remote MDP: Remote sends domain, client sends actions, remote sends state deltas
        *   - MDP stays on remote end
        *   - Would latency be too high?
        * Client MDP: Client/Remote specify MDP parameters, execution happens in client, client sends back complete trajectory
        *   - Duplicate MDP implementations need to match exactly
        *   - Useful for batch learning, sending back updated agent
     */
    socket.on('display', function (data: any) {
        //terrain = data
        //game.drawState()
    })
}
