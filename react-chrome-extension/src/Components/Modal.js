import React, { useEffect } from "react";
import axios from "axios";
import { X } from "react-feather";
import Draggable from "react-draggable";
import { ModalContext } from "../Contexts/ModalProvider";
const api = "http://127.0.0.1:5000/";

const Modal = () => {
  //   //this will be used to return from the backend the dictionary of image->date pairs
  //   const [imagePairs, search]

  // // Unique ID for the className.
  // var MOUSE_VISITED_CLASSNAME = "crx_mouse_visited";

  // // Previous dom, that we want to track, so we can remove the previous styling.
  // var prevDOM = null;

  // // Mouse listener for any move event on the current document.
  // document.addEventListener(
  //   "mousemove",
  //   function (e) {
  //     let srcElement = e.srcElement;

  //     // Lets check if our underlying element is a IMG.
  //     if (prevDOM != srcElement && srcElement.nodeName == "IMG") {
  //       // For NPE checking, we check safely. We need to remove the class name
  //       // Since we will be styling the new one after.
  //       if (prevDOM != null) {
  //         prevDOM.classList.remove(MOUSE_VISITED_CLASSNAME);
  //       }

  //       // Add a visited class name to the element. So we can style it.
  //       srcElement.classList.add(MOUSE_VISITED_CLASSNAME);

  //       // The current element is now the previous. So we can remove the class
  //       // during the next ieration.
  //       prevDOM = srcElement;
  //       console.info(srcElement.currentSrc);
  //       console.dir(srcElement);
  //     }
  //   },
  //   false
  // );

  const getURL = () => {
    return document.URL;
  };

  const callScript = async () => {
    const url = getURL();
    const response = await axios.get(
      "http://127.0.0.1:5000/geturl/?page=Paper_Mario:_The_Origami_King"
    );
    return response;
  };
  return (
    <ModalContext.Consumer>
      {({
        windowPosition,
        hasDraggedWindowPosition,
        extensionId,
        getExtensionId,
      }) => (
        <Draggable
          handle=".modal-handle"
          defaultPosition={{ x: windowPosition.x, y: windowPosition.y }}
          position={
            hasDraggedWindowPosition
              ? { x: windowPosition.x, y: windowPosition.y }
              : null
          }
        >
          <div
            id="modal"
            className="modal-window"
            style={{
              transform: windowPosition,
            }}
          >
            <div className="modal-window-inner-border">
              <>
                <div className="modal-body">
                  <div className="modal-handle">
                    <div className="modal-close-button">
                      <X color="#5d6484" size="14" />
                    </div>
                  </div>
                  <div className="modal-content">
                    <h3>{extensionId}</h3>

                    <button
                      onClick={() => {
                        console.log(getURL());

                        //console.log(callScript());
                      }}
                      className="modal-button"
                    >
                      RUN MINUTE MAN?
                    </button>
                  </div>
                </div>
              </>
            </div>
          </div>
        </Draggable>
      )}
    </ModalContext.Consumer>
  );
};

export default Modal;
