### Contents

All the code necessary for the Chrome extension is in the public folder

- imageMarker.css contains the css added after hovering over an image
- imageMarker.js contains the code for calling the backend API and alerting the user with the output description after hovering over an image
- manifest.json contains information about the extension

### Building The Chrome Extension

This project needs to be built in order to take advantage of the Chrome Extension API

To load as a developer extension inside of Chrome:

1. `npm run build` <br>
2. Navigate to `chrome://extensions/` in your browser <br>
3. Toggle the `Developer mode` switch on in the top right hand corner <br>
4. Click the `Load unpacked` button in the top left corner <br>
5. Select the `build` folder inside of this project folder <br>

Builds the app for Chrome to the `build` folder.<br>

