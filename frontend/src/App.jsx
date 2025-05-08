"use client";

import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import "./App.css";
import UploadFile from "./Pages/UploadFile";
import SelectColumns from "./Pages/SelectColumns";
import SelectKpi from "./Pages/SelectKPI";
import ClusteringComponent from "./Pages/ClusteringComponent";
import Home from "./Pages/Home";
import NoPage from "./Pages/NoPage";
import Projection from "./Pages/Projection";
import Workbench from "./Pages/Workbench";
import { Outlet, useLocation } from "react-router-dom";
import AppBar from "./Components/AppBar";
import Configuration from "./Pages/Configuration";
import AnalysisDashboard from "./Pages/AnalysisDashboard";
import Login from "./Pages/Login";
import Signup from "./Pages/Signup";
import { useEffect, useState } from "react";
import PropTypes from "prop-types";

// Protected Route component
const ProtectedRoute = ({ children }) => {
  const location = useLocation();
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    console.log("ProtectedRoute: Checking authentication");
    // Check if user is authenticated
    const session =
      localStorage.getItem("session") || sessionStorage.getItem("session");
    const com_id =
      localStorage.getItem("com_id") || sessionStorage.getItem("com_id");
    console.log("ProtectedRoute: Session found?", !!session);
    console.log("ProtectedRoute: Com ID found?", !!com_id);
    setIsAuthenticated(!!session && !!com_id);
    setLoading(false);
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        Loading...
      </div>
    );
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  return children;
};

ProtectedRoute.propTypes = {
  children: PropTypes.node.isRequired,
};

function Layout() {
  return (
    <>
      <AppBar />
      <div className="bg-gray-100 dark:bg-gray-900 min-h-[calc(100vh-64px)] h-[calc(100vh-64px)] overflow-hidden overflow-y-auto">
        <Outlet />
      </div>
    </>
  );
}

function App() {
  return (
    <>
      <BrowserRouter>
        <Routes>
          {/* Root route checks authentication and redirects appropriately */}
          <Route
            path="/"
            element={
              <ProtectedRoute>
                <Navigate to={`/${localStorage.getItem("com_id")}`} replace />
              </ProtectedRoute>
            }
          />

          {/* Public routes */}
          <Route path="/login" element={<Login />} />
          <Route path="/signup" element={<Signup />} />

          {/* Protected routes with com_id */}
          <Route
            path="/:com_id"
            element={
              <ProtectedRoute>
                <Layout />
              </ProtectedRoute>
            }
          >
            {/* Dashboard and analysis routes */}
            <Route index element={<Home />} />
            <Route path="analysis-1" element={<AnalysisDashboard />} />

            {/* Project routes */}
            <Route path="projects/:project_id">
              <Route index element={<Configuration />} />
              <Route path="upload" element={<UploadFile />} />
              <Route path="configuration" element={<Configuration />} />
              <Route path="analysis" element={<AnalysisDashboard />} />
              <Route path="select-columns" element={<SelectColumns />} />
              <Route path="select-kpi" element={<SelectKpi />} />
              <Route path="clustered-data" element={<ClusteringComponent />} />
              <Route path="workbench" element={<Workbench />} />
              <Route path="projection" element={<Projection />} />
            </Route>
          </Route>

          {/* Catch-all route */}
          <Route path="*" element={<NoPage />} />
        </Routes>
      </BrowserRouter>
    </>
  );
}

export default App;
