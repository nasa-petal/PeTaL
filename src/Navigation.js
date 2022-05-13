import { Navigate, Route, Routes } from "react-router-dom";
import HomePage from "./Pages/Home";

export default function Navigation() {
  return (
    <Routes>
      <Route element={<HomePage/>} index />
    </Routes>
  );
}