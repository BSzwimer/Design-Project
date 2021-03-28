async function fetchMovies() {
  const response = await fetch(
    "http://127.0.0.1:5000/geturl/?page=Paper_Mario:_The_Origami_King"
  );
  console.log(response);
}

fetchMovies();

var data = {};
function getData() {
  data = {
    "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c7/Suez_Canal_traffic_jam_seen_from_space.jpg/220px-Suez_Canal_traffic_jam_seen_from_space.jpg":
      "Eths tiny cock",
  };
}

// Unique ID for the className.

getData();
var MOUSE_VISITED_CLASSNAME = "crx_mouse_visited";

// Previous dom, that we want to track, so we can remove the previous styling.
var prevDOM = null;

// Mouse listener for any move event on the current document.
document.addEventListener(
  "mousemove",
  function (e) {
    let srcElement = e.srcElement;

    // Lets check if our underlying element is a IMG.
    if (prevDOM != srcElement && srcElement.nodeName == "IMG") {
      // For NPE checking, we check safely. We need to remove the class name
      // Since we will be styling the new one after.
      if (prevDOM != null) {
        prevDOM.classList.remove(MOUSE_VISITED_CLASSNAME);
      }

      // Add a visited class name to the element. So we can style it.
      srcElement.classList.add(MOUSE_VISITED_CLASSNAME);

      // The current element is now the previous. So we can remove the class
      // during the next ieration.
      prevDOM = srcElement;
      //console.info(srcElement.currentSrc);
      console.log(document.location.href);
      console.log(data[srcElement.currentSrc]);
      console.dir(srcElement);
    }
  },
  false
);
