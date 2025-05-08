"use client";

import { useEffect, useState } from "react";
import Plot from "react-plotly.js";
import { useLocation, useParams, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { ArrowLeft, Download, Share2, Settings, Loader2 } from "lucide-react";

const Projection = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { project_id, com_id } = useParams();

  const [plotData, setPlotData] = useState([]);
  const [plotLayout, setPlotLayout] = useState({});
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeKPI, setActiveKPI] = useState("Revenue"); // Default value, can be updated based on the actual KPI
  const [adjustments, setAdjustments] = useState({});

  // Animation variants
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        when: "beforeChildren",
        staggerChildren: 0.1,
      },
    },
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: { type: "spring", stiffness: 300, damping: 24 },
    },
  };

  useEffect(() => {
    setLoading(true);
    if (location.state && location.state.timeSeriesFigure) {
      try {
        const parsedData = JSON.parse(
          location.state.timeSeriesFigure.plotly_figure
        );
        setPlotData(parsedData.data || []);
        setPlotLayout(parsedData.layout || {});
        setError(null);
      } catch (err) {
        console.error("Error parsing time series figure:", err);
        setError("Failed to load chart data.");
      } finally {
        setLoading(false);
      }
    } else {
      setError("No chart data found. Please generate a projection first.");
      setLoading(false);
    }
  }, [location.state]);

  const handleDownload = () => {
    // This would typically download the chart as an image
    // For now, we'll just log a message
    console.log("Download functionality would be implemented here");
  };

  return (
    <motion.div
      className="min-h-screen bg-gray-50 dark:bg-gray-900 p-6"
      variants={containerVariants}
      initial="hidden"
      animate="visible"
    >
      <div className="max-w-7xl mx-auto">
        <motion.div
          variants={itemVariants}
          className="mb-6 flex flex-col sm:flex-row sm:items-center justify-between gap-4"
        >
          <div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
              Time Series Projection
            </h1>
            <p className="text-gray-600 dark:text-gray-400">
              Visualizing projected data for Project ID: {project_id}
            </p>
          </div>

          <div className="flex items-center gap-2">
            <motion.button
              onClick={() =>
                navigate(`/${com_id}/projects/${project_id}/workbench`)
              }
              className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-md hover:bg-gray-50 dark:hover:bg-gray-700"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <ArrowLeft className="h-4 w-4" />
              <span>Back</span>
            </motion.button>

            <motion.button
              onClick={handleDownload}
              className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-indigo-600 dark:text-indigo-400 bg-indigo-50 dark:bg-indigo-900/30 rounded-md hover:bg-indigo-100 dark:hover:bg-indigo-900/50"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              disabled={loading || error}
            >
              <Download className="h-4 w-4" />
              <span>Download</span>
            </motion.button>

            <motion.button
              className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-md hover:bg-gray-50 dark:hover:bg-gray-700"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              disabled={loading || error}
            >
              <Share2 className="h-4 w-4" />
              <span>Share</span>
            </motion.button>

            <motion.button
              className="flex items-center justify-center w-10 h-10 text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-md hover:bg-gray-50 dark:hover:bg-gray-700"
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              disabled={loading || error}
            >
              <Settings className="h-5 w-5" />
            </motion.button>
          </div>
        </motion.div>

        <motion.div
          variants={itemVariants}
          className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden"
        >
          {loading ? (
            <div className="flex flex-col items-center justify-center py-20">
              <Loader2 className="h-10 w-10 text-indigo-500 dark:text-indigo-400 animate-spin mb-4" />
              <p className="text-gray-600 dark:text-gray-400">
                Loading projection data...
              </p>
            </div>
          ) : error ? (
            <div className="flex flex-col items-center justify-center py-20 px-4 text-center">
              <div className="w-16 h-16 bg-red-100 dark:bg-red-900/30 rounded-full flex items-center justify-center mb-4">
                <Settings className="h-8 w-8 text-red-500 dark:text-red-400" />
              </div>
              <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                {error}
              </h3>
              <p className="text-gray-600 dark:text-gray-400 max-w-md mb-6">
                Please return to the workbench and generate a projection first.
              </p>
              <motion.button
                onClick={() =>
                  navigate(`/${com_id}/projects/${project_id}/workbench`)
                }
                className="flex items-center gap-2 px-6 py-2 text-sm font-medium text-white bg-indigo-600 rounded-md hover:bg-indigo-700"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <ArrowLeft className="h-4 w-4" />
                <span>Back to Workbench</span>
              </motion.button>
            </div>
          ) : (
            <div className="p-6">
              <Plot
                data={plotData}
                layout={{
                  ...plotLayout,
                  autosize: true,
                  width: undefined,
                  height: undefined,
                  paper_bgcolor: "rgba(0,0,0,0)",
                  plot_bgcolor: "rgba(0,0,0,0)",
                  font: {
                    color: document.documentElement.classList.contains("dark")
                      ? "#e5e7eb"
                      : "#374151",
                  },
                  margin: { l: 60, r: 40, t: 40, b: 60 },
                }}
                config={{ responsive: true }}
                style={{ width: "100%", height: "100%", minHeight: "600px" }}
              />

              <div className="mt-6 pt-6 border-t border-gray-200 dark:border-gray-700">
                <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
                  Projection Details
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="bg-gray-50 dark:bg-gray-900/50 p-4 rounded-lg">
                    <p className="text-sm text-gray-500 dark:text-gray-400">
                      Time Range
                    </p>
                    <p className="text-lg font-medium text-gray-900 dark:text-white">
                      12 Months
                    </p>
                  </div>
                  <div className="bg-gray-50 dark:bg-gray-900/50 p-4 rounded-lg">
                    <p className="text-sm text-gray-500 dark:text-gray-400">
                      KPI
                    </p>
                    <p className="text-lg font-medium text-gray-900 dark:text-white">
                      {activeKPI || "Revenue"}
                    </p>
                  </div>
                  <div className="bg-gray-50 dark:bg-gray-900/50 p-4 rounded-lg">
                    <p className="text-sm text-gray-500 dark:text-gray-400">
                      Modified Parameters
                    </p>
                    <p className="text-lg font-medium text-gray-900 dark:text-white">
                      {Object.keys(adjustments || {}).length || 0}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}
        </motion.div>
      </div>
    </motion.div>
  );
};

export default Projection;
