import "./gridworld-mdp"
import "./gridworld-game"
import {Direction, TerrainMap, textToTerrain} from "./gridworld-mdp";
import {GridworldGame} from "./gridworld-game";
import * as io from "socket.io-client"

let [n, s, e, w] = [Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST];

let terrain: {terrain: TerrainMap} = textToTerrain([
    'XXXXXXX',
    'X    GX',
    'X-----X',
    'X-----X',
    'X-----X',
    'X0----X',
    'XXXXXXX'
])
console.log(terrain)
let game = new GridworldGame(terrain["terrain"],
    document.getElementById("task")
)

game.init();
let atraj = [n, s, e, w]
let state = game.mdp.getStartState();

const start_i = 0
for (let i = 0; i < atraj.length; i++) {
    const statei = state.deepcopy();
    statei.agentPositions[0].x = i
    setTimeout(((state) => {
        return () => {
            //console.log(state)
            //game.drawState(state);
        }
    })(statei), 1000 + 750 * i);


}
function f() {
    const socket = io("http://127.0.0.1:5000");
    socket.on('connect', function() {
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
    socket.on('display', function(data: any) {
        terrain = data
        game.drawState()
    })
}
