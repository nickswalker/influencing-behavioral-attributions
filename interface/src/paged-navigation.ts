var current = 1; //the current page
let numPages = 0;
let pagesContainer: HTMLElement = null;
let pageNumber: HTMLElement = null;
let preNextValidator: (arg0: HTMLElement) => string = null

export function setUpPages(container: HTMLElement, nextValidator: (arg0: HTMLElement) => string) {
    pagesContainer = container
    preNextValidator = nextValidator
    pageNumber = document.getElementById("page-number")
    let pages: any = container.querySelectorAll(".page")

    // Give pages index IDs
    for (let i = 0; i < pages.length; i++) {
        // IDs can't start with a number. Some things break if they do
        pages[i].setAttribute("id", "p" + i.toString())
        pages[i].style.display = "none"
    }

    document.getElementById("p0").style.display = "block";
    current = 0;
    numPages = pages.length
}

//make one vanish and the other appear
function swap(vanish: number, appear: number) {
    (<HTMLElement>pagesContainer.querySelector("#p" + vanish)).style.display = "none";
    (<HTMLElement>document.querySelector("#p" + appear)).style.display = "";
}

//go to the next page
function next() {
    let validationResult = preNextValidator(pagesContainer.querySelector("#p" + current)) ?? ""
    if (validationResult !== "") {
        alert(validationResult)
        return;
    }
    if (current === numPages - 1) {
        return;
    }
    current++;
    pageNumber.innerText = (current + 1).toString();
    swap(current - 1, current);
    window.scrollTo(0, 0);
}

function back() {
    if (current === 0) {
        return;
    }
    current--;
    pageNumber.innerText = (current + 1).toString();
    swap(current + 1, current);
    window.scrollTo(0, 0);
}

declare global {
    interface Window {
        next: any
        back: any
    }
}

window.next = next
window.back = back