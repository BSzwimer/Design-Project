function fetchWebpage() {
  var path = window.location.pathname.split("/")
  var page = path[path.length -1]
  var url = "http://127.0.0.1:5000/geturl/?page=" + page

  //  Fetch the url which will return a dictionary of
  //  the captions of all the images on the webpage
  const response = fetch(
    url
  ).then(
    (response) => response.json()
  ).then((data) => {
    // When the data is received bind the event listener with the new data

    console.log("Success:", data);
    //getData();
    var MOUSE_VISITED_CLASSNAME = "crx_mouse_visited";

    // Previous dom, that we want to track, so we can remove the previous styling.
    var prevDOM = null;

    // Mouse listener for any move event on the current document.
    document.addEventListener(
      "mousemove",
      function (e) {
        let srcElement = e.srcElement;
        // Check if our underlying element is a IMG.
        if (prevDOM != srcElement && srcElement.nodeName == "IMG") {
          // For NPE checking, we check safely. We need to remove the class name
          // Since we will be styling the new one after.

          // Add a visited class name to the element. So we can style it.
          srcElement.classList.add(MOUSE_VISITED_CLASSNAME);

          // Alert the user of the caption
          console.log(document.location.href);
          alert(data["https:" + srcElement.getAttribute('src')])
          console.dir(srcElement);
        }
      },
      false
    );
  });
}

fetchWebpage()
