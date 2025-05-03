import React from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import AboutUs from "./AboutUs";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* make “/” render your about-us page */}
        <Route path="/" element={<AboutUs />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
