import "./gridworld-mdp"
import "./gridworld-game"
import "./gridworld-trajectory-display"
import "./gridworld-trajectory-player"
import {setUpPages} from "./paged-navigation";
import {Direction, TerrainMap, textToTerrain} from "./gridworld-mdp";
import {GridworldGame} from "./gridworld-game";


let terrain =  "'XXXXXXX',\
    'X    GX',\
        'X-----X',\
        'X--F--X',\
        'X-----X',\
        'X0----X',\
        'XXXXXXX'"
const batch = [{"trajectory": [0, 0, 0, 1, 0, 1, 1, 1, 0, 0], "return": 1.0, "entropy": 1.28819256228539}, {"trajectory": [1, 1, 1, 0, 1, 0, 0, 0, 0, 0], "return": 1.0, "entropy": 1.4103868378388813}, {"trajectory": [0, 0, 1, 1, 0, 1, 1, 0, 0, 0], "return": 1.0, "entropy": 1.4429867280989206}, {"trajectory": [1, 0, 1, 0, 0, 1, 1, 0, 0, 0], "return": 1.0, "entropy": 1.4672969161666547}, {"trajectory": [1, 0, 1, 0, 1, 0, 1, 0, 0, 0], "return": 1.0, "entropy": 1.4837686468584161}, {"trajectory": [0, 1, 0, 1, 1, 0, 0, 1, 0, 0], "return": 1.0, "entropy": 1.4917118291027376}, {"trajectory": [0, 0, 1, 0, 0, 1, 1, 1, 0, 0], "return": 1.0, "entropy": 1.5042453752190077}, {"trajectory": [0, 1, 1, 0, 0, 0, 1, 1, 0, 0], "return": 1.0, "entropy": 1.508412008076134}, {"trajectory": [0, 1, 0, 1, 1, 1, 0, 0, 0, 0], "return": 1.0, "entropy": 1.5094403920939814}, {"trajectory": [0, 1, 0, 1, 1, 0, 1, 0, 0, 0], "return": 1.0, "entropy": 1.5116626204309886}, {"trajectory": [1, 0, 1, 1, 0, 0, 0, 1, 0, 0], "return": 1.0, "entropy": 1.5310007481853964}, {"trajectory": [0, 0, 1, 0, 1, 0, 1, 1, 0, 0], "return": 1.0, "entropy": 1.5345223577036649}, {"trajectory": [0, 1, 1, 1, 1, 0, 0, 0, 0, 0], "return": 1.0, "entropy": 1.5415051673414226}, {"trajectory": [1, 0, 1, 1, 0, 1, 0, 0, 0, 0], "return": 1.0, "entropy": 1.5429626891631936}, {"trajectory": [0, 0, 1, 0, 1, 1, 1, 0, 0, 0], "return": 1.0, "entropy": 1.5528003944973403}, {"trajectory": [1, 0, 0, 1, 0, 0, 1, 1, 0, 0], "return": 1.0, "entropy": 1.553722847735663}, {"trajectory": [1, 0, 1, 1, 1, 0, 0, 0, 0, 0], "return": 1.0, "entropy": 1.5562122557494682}, {"trajectory": [1, 1, 1, 0, 0, 0, 0, 1, 0, 0], "return": 1.0, "entropy": 1.568979224326843}, {"trajectory": [1, 0, 0, 1, 1, 0, 0, 1, 0, 0], "return": 1.0, "entropy": 1.5695674185000275}, {"trajectory": [0, 1, 0, 0, 0, 1, 1, 1, 0, 0], "return": 1.0, "entropy": 1.569603172834871}, {"trajectory": [0, 1, 1, 0, 0, 1, 1, 0, 0, 0], "return": 1.0, "entropy": 1.585197468439806}, {"trajectory": [1, 1, 0, 0, 1, 1, 0, 0, 0, 0], "return": 1.0, "entropy": 1.5874618600147672}, {"trajectory": [1, 0, 0, 0, 1, 1, 0, 1, 0, 0], "return": 1.0, "entropy": 1.5876716344145083}, {"trajectory": [1, 0, 1, 0, 0, 1, 0, 1, 0, 0], "return": 1.0, "entropy": 1.5981144795743414}, {"trajectory": [1, 1, 0, 1, 0, 0, 0, 1, 0, 0], "return": 1.0, "entropy": 1.6039400902547996}, {"trajectory": [0, 0, 0, 1, 1, 0, 1, 1, 0, 0], "return": 1.0, "entropy": 1.6114582494765612}, {"trajectory": [1, 1, 0, 1, 1, 0, 0, 0, 0, 0], "return": 1.0, "entropy": 1.6124188749237782}, {"trajectory": [0, 0, 0, 0, 1, 1, 1, 1, 0, 0], "return": 1.0, "entropy": 1.6129756838729739}, {"trajectory": [1, 1, 0, 1, 0, 0, 1, 0, 0, 0], "return": 1.0, "entropy": 1.6157132356012318}, {"trajectory": [0, 0, 1, 1, 0, 0, 1, 1, 0, 0], "return": 1.0, "entropy": 1.6249192162450719}, {"trajectory": [1, 0, 1, 0, 1, 1, 0, 0, 0, 0], "return": 1.0, "entropy": 1.625005873190119}, {"trajectory": [1, 1, 0, 1, 0, 1, 0, 0, 0, 0], "return": 1.0, "entropy": 1.6308618088625422}, {"trajectory": [1, 1, 0, 0, 0, 1, 0, 1, 0, 0], "return": 1.0, "entropy": 1.6323353113161425}, {"trajectory": [1, 0, 0, 1, 0, 1, 1, 0, 0, 0], "return": 1.0, "entropy": 1.6396534555747713}, {"trajectory": [0, 0, 0, 1, 1, 1, 1, 0, 0, 0], "return": 1.0, "entropy": 1.6589588107988238}, {"trajectory": [1, 0, 0, 1, 1, 0, 1, 0, 0, 0], "return": 1.0, "entropy": 1.6631103465663835}, {"trajectory": [0, 0, 1, 1, 0, 1, 0, 1, 0, 0], "return": 1.0, "entropy": 1.6640167792703218}, {"trajectory": [0, 1, 0, 1, 0, 0, 1, 1, 0, 0], "return": 1.0, "entropy": 1.6658285588016128}, {"trajectory": [1, 0, 0, 0, 1, 0, 1, 1, 0, 0], "return": 1.0, "entropy": 1.6785099116656925}, {"trajectory": [0, 1, 1, 1, 0, 1, 0, 0, 0, 0], "return": 1.0, "entropy": 1.6827511438296991}, {"trajectory": [1, 1, 0, 0, 0, 1, 1, 0, 0, 0], "return": 1.0, "entropy": 1.6839341123333333}, {"trajectory": [1, 1, 1, 0, 0, 1, 0, 0, 0, 0], "return": 1.0, "entropy": 1.6915839511238013}, {"trajectory": [0, 1, 0, 0, 1, 1, 0, 1, 0, 0], "return": 1.0, "entropy": 1.693570596331061}, {"trajectory": [0, 0, 1, 1, 1, 0, 1, 0, 0, 0], "return": 1.0, "entropy": 1.69713223469606}, {"trajectory": [1, 0, 0, 1, 0, 1, 0, 1, 0, 0], "return": 1.0, "entropy": 1.7208133125381964}, {"trajectory": [0, 1, 1, 0, 0, 1, 0, 1, 0, 0], "return": 1.0, "entropy": 1.7246528804784558}, {"trajectory": [0, 0, 1, 0, 1, 1, 0, 1, 0, 0], "return": 1.0, "entropy": 1.7261735888985332}, {"trajectory": [1, 0, 0, 0, 0, 1, 1, 1, 0, 0], "return": 1.0, "entropy": 1.7304793747470928}, {"trajectory": [0, 1, 1, 1, 0, 0, 1, 0, 0, 0], "return": 1.0, "entropy": 1.7347178349859655}, {"trajectory": [1, 0, 1, 0, 0, 0, 1, 1, 0, 0], "return": 1.0, "entropy": 1.7351672740702968}, {"trajectory": [0, 0, 0, 1, 1, 1, 0, 1, 0, 0], "return": 1.0, "entropy": 1.7363869894869706}, {"trajectory": [0, 1, 0, 1, 0, 1, 1, 0, 0, 0], "return": 1.0, "entropy": 1.7408609039138916}, {"trajectory": [0, 0, 1, 1, 1, 0, 0, 1, 0, 0], "return": 1.0, "entropy": 1.7413396528758762}, {"trajectory": [0, 1, 1, 0, 1, 1, 0, 0, 0, 0], "return": 1.0, "entropy": 1.7470198890092876}, {"trajectory": [0, 1, 1, 1, 0, 0, 0, 1, 0, 0], "return": 1.0, "entropy": 1.7556025872560088}, {"trajectory": [0, 0, 1, 1, 1, 1, 0, 0, 0, 0], "return": 1.0, "entropy": 1.7616303991927371}, {"trajectory": [1, 0, 0, 1, 1, 1, 0, 0, 0, 0], "return": 1.0, "entropy": 1.7670171873471079}, {"trajectory": [1, 0, 1, 0, 1, 0, 0, 1, 0, 0], "return": 1.0, "entropy": 1.7743825131796358}, {"trajectory": [0, 1, 0, 0, 1, 0, 1, 1, 0, 0], "return": 1.0, "entropy": 1.7842840840165064}, {"trajectory": [1, 1, 0, 0, 0, 0, 1, 1, 0, 0], "return": 1.0, "entropy": 1.8143181437363167}, {"trajectory": [1, 1, 1, 0, 0, 0, 1, 0, 0, 0], "return": 1.0, "entropy": 1.8212448853049747}, {"trajectory": [1, 1, 0, 0, 1, 0, 0, 1, 0, 0], "return": 1.0, "entropy": 1.8229807729232566}, {"trajectory": [1, 0, 1, 1, 0, 0, 1, 0, 0, 0], "return": 1.0, "entropy": 1.8259282589986072}, {"trajectory": [1, 1, 0, 0, 1, 0, 1, 0, 0, 0], "return": 1.0, "entropy": 1.8480173747213953}, {"trajectory": [0, 1, 1, 0, 1, 0, 1, 0, 0, 0], "return": 1.0, "entropy": 1.8687319246845342}, {"trajectory": [1, 1, 1, 1, 0, 0, 0, 0, 0, 0], "return": 1.0, "entropy": 1.8900492266796354}, {"trajectory": [0, 1, 0, 1, 0, 1, 0, 1, 0, 0], "return": 1.0, "entropy": 1.9100732226126722}, {"trajectory": [0, 1, 1, 0, 1, 0, 0, 1, 0, 0], "return": 1.0, "entropy": 1.9332037742014592}, {"trajectory": [0, 1, 0, 0, 1, 1, 1, 0, 0, 0], "return": 1.0, "entropy": 1.9819690384145807}, {"trajectory": [1, 0, 0, 0, 1, 1, 1, 0, 0, 0], "return": 1.0, "entropy": 1.9910898260371859}]


const unsorted = document.getElementById("unsorted")
const columnsContainer = document.getElementById("columns")
let numColumns = 0;
function dragstartHandler(ev: any) {
    // Add the target element's id to the data transfer object
    ev.dataTransfer.setData("text/plain", ev.currentTarget.id);
}
function dragoverHandler(ev: any) {
    ev.preventDefault();
    ev.dataTransfer.dropEffect = "move";
}
function dropHandler(ev: any) {
    ev.preventDefault();
    // Get the id of the target and add the moved element to the target's DOM
    const data = ev.dataTransfer.getData("text/plain");
    // We'll get child elements as targets. CurrentTarget is the bubbled target
    ev.currentTarget.appendChild(document.getElementById(data));
}
function addColumn(event: Event) {
    event?.preventDefault()
    const newColumn = document.createElement("ul")
    newColumn.classList.add("column")
    newColumn.id = "c" + numColumns
    newColumn.addEventListener("drop", dropHandler)
    newColumn.addEventListener("dragover", dragoverHandler)
    columnsContainer.appendChild(newColumn)
    numColumns+= 1;
}
// Put a few columns in by default
for (let i = 0; i < 4; i++) {
    addColumn(null)
}

for (let i = 0; i < batch.length; i++) {
    const list_item = document.createElement("li")
    list_item.id = "trajectory" + i
    const new_element = document.createElement("gridworld-display")
    //const new_element = document.createElement("div")
    //new_element.classList.add("stand-in")
    new_element.setAttribute("stepwise", "false")
    new_element.setAttribute("terrain", terrain)
    let traj_string = batch[i]["trajectory"].join(", ")
    new_element.setAttribute("trajectory", traj_string)

    list_item.setAttribute("draggable", "true")
    list_item.addEventListener("dragstart", dragstartHandler)
    list_item.appendChild(new_element)
    unsorted.appendChild(list_item)
}
const addColumnButton = document.getElementById("add-column")
addColumnButton.addEventListener("click", addColumn)

function updateLabeling(sortingPage: HTMLElement,  explanationsPage: HTMLElement) {
const columns = sortingPage.querySelectorAll(".column")
    // Clear out whatever was there before
    explanationsPage.innerHTML = ""
    columns.forEach((column: HTMLElement) => {
        const groupId = parseInt(column.id.slice(1))
        const elements = column.querySelectorAll("li")
        // Allow empty groups
        if (elements.length == 0) {
            return;
        }
        const group = document.createElement("div")
        group.classList.add("group-block")
        const list = document.createElement("ul")
        list.classList.add("group")
        group.appendChild(list)

        elements.forEach((element: HTMLElement) => {
            const elementCopy = element.cloneNode(true)
            list.appendChild(elementCopy)
        })
        const labelField = document.createElement("crowd-input")
        labelField.setAttribute("placeholder", "Why do these items belong together?")
        labelField.id = "explanation" + groupId
        labelField.setAttribute("name", "explanation" + groupId)
        group.appendChild(labelField)
        const groupMembers = document.createElement("input")
        groupMembers.name = "members"+groupId
        groupMembers.style.display = "none"
        elements.forEach((element) => {groupMembers.value += "," + element.id})
        group.appendChild(groupMembers)
        explanationsPage.appendChild(group)
    })

}

function validateNext(page: HTMLElement) : string {
    if (page.classList.contains("sorting")) {
        let unsorted = page.querySelectorAll("#unsorted li")
        if(unsorted.length > 0) {
            return "Please finish sorting"
        } else {
            updateLabeling(page, <HTMLElement>document.getElementsByClassName("explanations")[0])
        }
    }
    else if (page.classList.contains("qualification")) {
        let radioGroup = page.querySelector("crowd-radio-group");
        let options = radioGroup.querySelectorAll("crowd-radio-button")
        let rowValid = false;

        for (let i = 0; i < options.length; i++) {
            let option: HTMLInputElement = <HTMLInputElement>options[i]
            if (option.checked) rowValid = true;
            if (i === 0 && option.checked) {
                return "You are not eligible for this HIT. Please return it."
            }
        }

        if (!rowValid){
            return "Please select an answer"
        }

    }
    return null
}
setUpPages(document.getElementById("task"), validateNext)