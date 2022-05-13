import { Navigate, Route, Routes } from "react-router-dom";
import HomePage from "./Pages/Home";

export default function Navigation() {
  return (
    <Routes>
      <Route element={<HomePage/>} index />
      {/* <Route path="*" element={<Navigate replace to="/"/>} /> */}
    </Routes>
  );
}