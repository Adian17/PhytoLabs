import { createBrowserRouter } from "react-router";
import { RootLayout } from "./components/RootLayout";
import { Home } from "./components/screens/Home";
import { Capture } from "./components/screens/Capture";
import { Confirm } from "./components/screens/Confirm";
import { Analyzing } from "./components/screens/Analyzing";
import { Results } from "./components/screens/Results";
import { Treatment } from "./components/screens/Treatment";

export const router = createBrowserRouter([
  {
    path: "/",
    Component: RootLayout,
    children: [
      { index: true, Component: Home },
      { path: "capture", Component: Capture },
      { path: "confirm", Component: Confirm },
      { path: "analyzing", Component: Analyzing },
      { path: "results", Component: Results },
      { path: "treatment", Component: Treatment },
    ],
  },
]);
