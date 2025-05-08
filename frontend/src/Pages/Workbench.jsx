"use client";

import { useSelector } from "react-redux";
import React, { useState } from "react";
import axios from "axios";
import { useLocation, useNavigate, useParams } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import {
  ArrowRight,
  Loader2,
  Calendar,
  BarChart4,
  TrendingUp,
  Settings,
  Download,
  X,
  Info,
  ChevronDown,
  ChevronUp,
  Check,
  ChevronLeft,
} from "lucide-react";
import Plot from "react-plotly.js";

const Workbench = () => {
  const { clusterHistory, selectedIndex } = useSelector(
    (state) => state.cluster
  );
  const currentLevel = clusterHistory[selectedIndex]?.level;
  const location = useLocation();
  const navigate = useNavigate();
  const { activeKPI, kpiList, task_id, importantColumnNames } =
    location.state || {};
  const { project_id, com_id } = useParams();
  const [adjustments, setAdjustments] = useState({});
  const [noOfMonths, setNoOfMonths] = useState(12);
  const [dateColumn, setDateColumn] = useState("");
  const [weightData, setWeightData] = useState({});
  const [loadingIndex, setLoadingIndex] = useState(null);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const columns = JSON.parse(localStorage.getItem("columns") || "[]");
  const baseUrl = import.meta.env.VITE_BASE_URL;

  const regressors = Object.values(weightData)
    ?.flat()
    ?.map((item) => item.feature);

  const [isPopupVisible, setIsPopupVisible] = useState(false);
  const [plotData, setPlotData] = useState([]);
  const [plotLayout, setPlotLayout] = useState({});
  const [isLoading, setIsLoading] = useState(false);
  const [checkedState, setCheckedState] = React.useState(
    new Array(clusterHistory.length).fill(false)
  );
  const [childCheckedState, setChildCheckedState] = React.useState({});
  const [expandedSections, setExpandedSections] = useState({});

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

  const popupVariants = {
    hidden: { opacity: 0, scale: 0.95 },
    visible: {
      opacity: 1,
      scale: 1,
      transition: { type: "spring", damping: 25, stiffness: 300 },
    },
  };

  const handleActionClick = async () => {
    // Validate required fields
    if (!dateColumn) {
      setError("Please select a date column");
      setSuccess("");
      return;
    }
    if (!noOfMonths || noOfMonths <= 0) {
      setError(
        "Please enter a valid number of months and it should be greater than 0"
      );
      setSuccess("");
      return;
    }
    if (!checkedState.some((state) => state)) {
      setError("Please select at least one checkbox");
      setSuccess("");
      return;
    }
    setError("");
    setIsLoading(true);
    setSuccess("");

    try {
      // Initial API call
      const response = await axios.post(
        `${baseUrl}/projects/${project_id}/time-series/analysis`,
        {
          path: clusterHistory[0].path,
          kpi: activeKPI,
          no_of_months: Number(noOfMonths),
          date_column: dateColumn,
          adjustments: adjustments,
          regressors: regressors,
        },
        {
          headers: {
            "Content-Type": "application/json",
          },
        }
      );

      const { task_id } = response.data;

      // Function to check task status
      const checkStatus = async () => {
        try {
          const statusResponse = await axios.get(
            `${baseUrl}/projects/tasks/${task_id}/status`,
            {
              headers: {
                "Content-Type": "application/json",
              },
            }
          );
          return statusResponse.data.status;
        } catch (error) {
          console.error("Error checking status:", error);
          throw error;
        }
      };

      // Poll status until SUCCESS
      const pollStatus = async () => {
        return new Promise((resolve, reject) => {
          const interval = setInterval(async () => {
            try {
              const status = await checkStatus();
              if (status === "SUCCESS") {
                clearInterval(interval);
                // Get the analysis results
                const analysisResponse = await axios.post(
                  `${baseUrl}/projects/${project_id}/time-series/figure`,
                  {
                    kpi: activeKPI,
                    path: clusterHistory[0].path,
                  },
                  {
                    headers: {
                      "Content-Type": "application/json",
                    },
                  }
                );
                setPlotData(
                  JSON.parse(analysisResponse.data.plotly_figure).data
                );
                setPlotLayout(
                  JSON.parse(analysisResponse.data.plotly_figure).layout
                );
                setIsPopupVisible(true);
                setSuccess("Time series analysis completed successfully!");
                resolve(analysisResponse.data.plotly_figure);
              } else if (status === "FAILURE") {
                clearInterval(interval);
                setError("Analysis task failed. Please try again.");
                reject(new Error("Analysis task failed"));
              }
            } catch (error) {
              clearInterval(interval);
              setError("Error in time series analysis. Please try again.");
              reject(error);
            }
          }, 1000); // Check every second
        });
      };

      // Start polling and wait for results
      await pollStatus();
    } catch (error) {
      console.error("Error in time series analysis:", error);
      setError("Failed to complete time series analysis. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const closePopup = () => {
    setIsPopupVisible(false);
  };

  const handleCheckboxChange = (index, isChild = false, parentIndex = null) => {
    if (isChild) {
      // Handle child checkbox
      const newChildCheckedState = {
        ...childCheckedState,
        [parentIndex]: {
          ...childCheckedState[parentIndex],
          [index]: !childCheckedState[parentIndex]?.[index],
        },
      };
      setChildCheckedState(newChildCheckedState);

      const feature = weightData[parentIndex][index].feature;
      setAdjustments((prev) => {
        const newAdjustments = { ...prev };
        if (!childCheckedState[parentIndex]?.[index]) {
          // If it's being checked, add an empty value
          newAdjustments[feature] = "";
        } else {
          // If it's being unchecked, remove the value
          delete newAdjustments[feature];
        }
        return newAdjustments;
      });
    } else {
      // Handle parent checkbox
      const updatedCheckedState = [...checkedState];
      updatedCheckedState[index] = !updatedCheckedState[index];
      setCheckedState(updatedCheckedState);

      // Clear adjustments for this parent if unchecked
      if (!updatedCheckedState[index]) {
        const feature =
          typeof clusterHistory[index].value === "number"
            ? clusterHistory[index].feature
            : clusterHistory[index].value;

        setAdjustments((prev) => {
          const newAdjustments = { ...prev };
          delete newAdjustments[feature];

          // Also clear child adjustments if they exist
          if (weightData[index]) {
            weightData[index].forEach((item) => {
              delete newAdjustments[item.feature];
            });
          }
          return newAdjustments;
        });

        // Clear child checked states
        setChildCheckedState((prev) => {
          const newState = { ...prev };
          delete newState[index];
          return newState;
        });

        // Collapse the section
        setExpandedSections((prev) => {
          const newState = { ...prev };
          delete newState[index];
          return newState;
        });
      }
    }
  };

  const handleInputChange = (
    index,
    value,
    isChild = false,
    parentIndex = null
  ) => {
    let feature;
    if (isChild) {
      feature = weightData[parentIndex][index].feature;
    } else {
      feature =
        typeof clusterHistory[index].value === "number"
          ? clusterHistory[index].feature
          : clusterHistory[index].value;
    }

    setAdjustments((prev) => ({
      ...prev,
      [feature]: value,
    }));
  };

  const handleOneHotEncoding = async (index) => {
    setLoadingIndex(index);
    setError("");
    setSuccess("");

    try {
      // Initial API call
      const response = await axios.post(
        `${baseUrl}/projects/${project_id}/features/onehot`,
        {
          path: clusterHistory[index].path,
          kpi:
            typeof clusterHistory[index].value === "number"
              ? clusterHistory[index].feature
              : clusterHistory[index].value,
        }
      );

      const { task_id } = response.data;

      // Function to check task status
      const checkStatus = async () => {
        try {
          const statusResponse = await axios.get(
            `${baseUrl}/projects/tasks/${task_id}/status`,
            {
              headers: {
                "Content-Type": "application/json",
              },
            }
          );

          return statusResponse.data.status;
        } catch (error) {
          console.error("Error checking status:", error);
          throw error;
        }
      };

      // Poll status until SUCCESS
      const pollStatus = async () => {
        return new Promise((resolve, reject) => {
          const interval = setInterval(async () => {
            try {
              const status = await checkStatus();
              if (status === "SUCCESS") {
                clearInterval(interval);
                // Make the final API call
                const weightResponse = await axios.post(
                  `${baseUrl}/projects/${project_id}/features/weight/onehot`,
                  {
                    path: clusterHistory[index].path,
                    kpi:
                      typeof clusterHistory[index].value === "number"
                        ? clusterHistory[index].feature
                        : clusterHistory[index].value,
                  },
                  {
                    headers: {
                      "Content-Type": "application/json",
                    },
                  }
                );
                setWeightData((prevState) => ({
                  ...prevState,
                  [index]: weightResponse.data.slice(0, 5),
                }));

                // Expand the section
                setExpandedSections((prev) => ({
                  ...prev,
                  [index]: true,
                }));

                setSuccess("One-hot encoding completed successfully!");
                resolve(weightResponse.data);
              } else if (status === "FAILURE") {
                clearInterval(interval);
                setError("One-hot encoding task failed. Please try again.");
                reject(new Error("Task failed"));
              }
            } catch (error) {
              clearInterval(interval);
              setError("Error in one-hot encoding process. Please try again.");
              reject(error);
            }
          }, 1000); // Check every second
        });
      };

      // Start polling
      await pollStatus();
    } catch (error) {
      console.error("Error in OneHot encoding process:", error);
      setError("Failed to complete one-hot encoding. Please try again.");
    } finally {
      setLoadingIndex(null);
    }
  };

  const toggleSection = (index) => {
    setExpandedSections((prev) => ({
      ...prev,
      [index]: !prev[index],
    }));
  };

  return (
    <motion.div
      className="bg-gray-50 dark:bg-gray-900 min-h-[calc(100vh-64px)]"
      variants={containerVariants}
      initial="hidden"
      animate="visible"
    >
      <div className="container mx-auto px-4 py-8">
        {/* Add back navigation button in the header section */}
        <motion.div variants={itemVariants} className="mb-8">
          <div className="flex items-center gap-3">
            <motion.button
              onClick={() =>
                navigate(`/${com_id}/projects/${project_id}/clustered-data`)
              }
              className="p-2 rounded-full bg-white dark:bg-gray-800 shadow-sm border border-gray-200 dark:border-gray-700"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <ChevronLeft className="h-5 w-5 text-gray-600 dark:text-gray-400" />
            </motion.button>
            <div>
              <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
                Workbench
              </h1>
              <p className="text-gray-600 dark:text-gray-400">
                Analyze and modify parameters to generate time series
                projections
              </p>
            </div>
          </div>
        </motion.div>

        {/* Cluster History Timeline */}
        <motion.div
          variants={itemVariants}
          className="mb-8 bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6"
        >
          <h2 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4 flex items-center gap-2">
            <Calendar className="h-5 w-5 text-indigo-500 dark:text-indigo-400" />
            Segment History
          </h2>

          <div className="overflow-x-auto">
            <div className="flex items-center space-x-4 min-w-max">
              {clusterHistory.map((cluster, index) => (
                <React.Fragment key={index}>
                  <motion.div
                    whileHover={{ y: -5 }}
                    className="flex-shrink-0 w-64 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden"
                  >
                    <div className="h-24 bg-gradient-to-r from-indigo-500 to-purple-600 relative overflow-hidden">
                      <div className="absolute inset-0 flex items-center justify-center">
                        <BarChart4 className="h-12 w-12 text-white/20" />
                      </div>
                      <div className="absolute bottom-0 left-0 right-0 p-3 bg-gradient-to-t from-black/50 to-transparent">
                        <p className="text-white font-bold">
                          Level {(cluster.level || 0) + 1}
                        </p>
                      </div>
                    </div>
                    <div className="p-4 bg-white dark:bg-gray-800">
                      <div className="space-y-2">
                        <p className="text-sm flex justify-between">
                          <span className="text-gray-500 dark:text-gray-400">
                            Value:
                          </span>
                          <span className="font-medium text-gray-900 dark:text-gray-100">
                            {typeof cluster.value === "number"
                              ? cluster.value.toFixed(4)
                              : `${cluster.value} - ${cluster.percentage}%`}
                          </span>
                        </p>
                        <p className="text-sm flex justify-between">
                          <span className="text-gray-500 dark:text-gray-400">
                            Parameter:
                          </span>
                          <span className="font-medium text-gray-900 dark:text-gray-100 truncate max-w-[120px]">
                            {cluster.feature || "Parameter"}
                          </span>
                        </p>
                        <p className="text-sm flex justify-between">
                          <span className="text-gray-500 dark:text-gray-400">
                            Segment:
                          </span>
                          <span className="font-medium text-gray-900 dark:text-gray-100">
                            {cluster.cluster}
                          </span>
                        </p>
                      </div>
                    </div>
                  </motion.div>
                  {index < clusterHistory.length - 1 && (
                    <div className="flex-shrink-0">
                      <ArrowRight className="h-6 w-6 text-gray-400 dark:text-gray-600" />
                    </div>
                  )}
                </React.Fragment>
              ))}
            </div>
          </div>
        </motion.div>

        {/* Definition Table */}
        <motion.div
          variants={itemVariants}
          className="mb-8 bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden"
        >
          <div className="p-4 border-b border-gray-200 dark:border-gray-700">
            <h2 className="text-lg font-semibold text-gray-800 dark:text-gray-200 flex items-center gap-2">
              <Settings className="h-5 w-5 text-indigo-500 dark:text-indigo-400" />
              Parameter Adjustments
            </h2>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full border-collapse">
              <thead>
                <tr className="bg-gray-50 dark:bg-gray-700/50">
                  <th className="py-3 px-4 text-left text-sm font-medium text-gray-700 dark:text-gray-300 border-b border-gray-200 dark:border-gray-700"></th>
                  <th className="py-3 px-4 text-left text-sm font-medium text-gray-700 dark:text-gray-300 border-b border-gray-200 dark:border-gray-700">
                    Parameter
                  </th>
                  <th className="py-3 px-4 text-left text-sm font-medium text-gray-700 dark:text-gray-300 border-b border-gray-200 dark:border-gray-700">
                    Current Value
                  </th>
                  <th className="py-3 px-4 text-left text-sm font-medium text-gray-700 dark:text-gray-300 border-b border-gray-200 dark:border-gray-700">
                    Modify To
                  </th>
                </tr>
              </thead>
              <tbody>
                {clusterHistory.map((cluster, index) => (
                  <React.Fragment key={index}>
                    <tr className="border-b border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700/30">
                      <td className="py-3 px-4">
                        <div className="flex items-center">
                          <input
                            type="checkbox"
                            checked={checkedState[index]}
                            onChange={() => handleCheckboxChange(index)}
                            className="h-4 w-4 rounded border-gray-300 text-indigo-600 focus:ring-indigo-500 dark:border-gray-600 dark:bg-gray-700 dark:focus:ring-indigo-400"
                            disabled={isLoading}
                          />
                        </div>
                      </td>
                      <td className="py-3 px-4 font-medium text-gray-900 dark:text-white">
                        {cluster.feature || "Parameter"}
                      </td>
                      <td className="py-3 px-4 text-gray-700 dark:text-gray-300">
                        {typeof cluster.value === "number"
                          ? cluster.value.toFixed(4)
                          : `${cluster.value} - ${cluster.percentage}%`}
                      </td>
                      <td className="py-3 px-4">
                        <div className="flex items-center gap-2">
                          <input
                            className={`border rounded px-3 py-1.5 w-full max-w-xs text-gray-900 dark:text-gray-100 bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-600 focus:ring-indigo-500 dark:focus:ring-indigo-400 focus:border-indigo-500 dark:focus:border-indigo-400 ${
                              !checkedState[index]
                                ? "bg-gray-100 dark:bg-gray-700 cursor-not-allowed"
                                : ""
                            }`}
                            disabled={!checkedState[index] || isLoading}
                            onChange={(e) =>
                              handleInputChange(index, e.target.value)
                            }
                            value={
                              adjustments[
                                typeof cluster.value === "number"
                                  ? cluster.feature
                                  : cluster.value
                              ] || ""
                            }
                            placeholder="New value"
                          />
                          {checkedState[index] && (
                            <div className="flex items-center gap-2">
                              <motion.button
                                onClick={() => handleOneHotEncoding(index)}
                                className={`h-8 px-2 rounded-md flex items-center gap-1 text-xs font-medium ${
                                  loadingIndex === index
                                    ? "bg-gray-200 dark:bg-gray-700 text-gray-500 dark:text-gray-400 cursor-not-allowed"
                                    : "bg-indigo-50 dark:bg-indigo-900/30 text-indigo-600 dark:text-indigo-400 hover:bg-indigo-100 dark:hover:bg-indigo-900/50"
                                }`}
                                title="Generate OneHot Encoding"
                                disabled={loadingIndex === index || isLoading}
                                whileHover={
                                  loadingIndex === index ? {} : { scale: 1.05 }
                                }
                                whileTap={
                                  loadingIndex === index ? {} : { scale: 0.95 }
                                }
                              >
                                {loadingIndex === index ? (
                                  <Loader2 className="w-3 h-3 animate-spin" />
                                ) : (
                                  <>
                                    <TrendingUp className="w-3 h-3" />
                                    <span>Encode</span>
                                  </>
                                )}
                              </motion.button>

                              {weightData[index] && (
                                <motion.button
                                  onClick={() => toggleSection(index)}
                                  className="h-8 w-8 rounded-md flex items-center justify-center text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700"
                                  whileHover={{ scale: 1.1 }}
                                  whileTap={{ scale: 0.9 }}
                                >
                                  {expandedSections[index] ? (
                                    <ChevronUp className="w-4 h-4" />
                                  ) : (
                                    <ChevronDown className="w-4 h-4" />
                                  )}
                                </motion.button>
                              )}
                            </div>
                          )}
                        </div>
                      </td>
                    </tr>

                    {/* Sub-parameters section */}
                    {weightData[index] &&
                      checkedState[index] &&
                      expandedSections[index] && (
                        <tr>
                          <td
                            colSpan="4"
                            className="py-0 px-0 border-b border-gray-200 dark:border-gray-700"
                          >
                            <AnimatePresence>
                              <motion.div
                                initial={{ opacity: 0, height: 0 }}
                                animate={{ opacity: 1, height: "auto" }}
                                exit={{ opacity: 0, height: 0 }}
                                transition={{ duration: 0.2 }}
                                className="bg-gray-50 dark:bg-gray-800/50 px-4 py-3"
                              >
                                <div className="mb-2 flex items-center justify-between">
                                  <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300">
                                    Sub-parameters
                                  </h3>
                                  <span className="text-xs text-gray-500 dark:text-gray-400">
                                    Select parameters to adjust
                                  </span>
                                </div>
                                <table className="w-full border-collapse">
                                  <thead>
                                    <tr>
                                      <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400">
                                        Select
                                      </th>
                                      <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400">
                                        Feature
                                      </th>
                                      <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400">
                                        Modify To
                                      </th>
                                    </tr>
                                  </thead>
                                  <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                                    {weightData[index].map((item, idx) => {
                                      const isChecked =
                                        childCheckedState[index]?.[idx] ||
                                        false;
                                      return (
                                        <motion.tr
                                          key={idx}
                                          className="hover:bg-gray-100 dark:hover:bg-gray-700/30"
                                          initial={{ opacity: 0, y: 10 }}
                                          animate={{ opacity: 1, y: 0 }}
                                          transition={{ delay: idx * 0.05 }}
                                        >
                                          <td className="px-4 py-2">
                                            <input
                                              type="checkbox"
                                              checked={isChecked}
                                              onChange={() =>
                                                handleCheckboxChange(
                                                  idx,
                                                  true,
                                                  index
                                                )
                                              }
                                              className="h-4 w-4 rounded border-gray-300 text-indigo-600 focus:ring-indigo-500 dark:border-gray-600 dark:bg-gray-700 dark:focus:ring-indigo-400"
                                              disabled={isLoading}
                                            />
                                          </td>
                                          <td className="px-4 py-2 text-sm text-gray-700 dark:text-gray-300">
                                            {item.feature || "-"}
                                          </td>
                                          <td className="px-4 py-2">
                                            <input
                                              className={`border rounded px-3 py-1.5 w-full max-w-xs text-gray-900 dark:text-gray-100 bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-600 focus:ring-indigo-500 dark:focus:ring-indigo-400 focus:border-indigo-500 dark:focus:border-indigo-400 ${
                                                !isChecked
                                                  ? "bg-gray-100 dark:bg-gray-700 cursor-not-allowed"
                                                  : ""
                                              }`}
                                              disabled={!isChecked || isLoading}
                                              value={
                                                adjustments[item.feature] || ""
                                              }
                                              onChange={(e) =>
                                                handleInputChange(
                                                  idx,
                                                  e.target.value,
                                                  true,
                                                  index
                                                )
                                              }
                                              placeholder="New value"
                                            />
                                          </td>
                                        </motion.tr>
                                      );
                                    })}
                                  </tbody>
                                </table>
                              </motion.div>
                            </AnimatePresence>
                          </td>
                        </tr>
                      )}
                  </React.Fragment>
                ))}
              </tbody>
            </table>
          </div>
        </motion.div>

        {/* Configuration Section */}
        <motion.div
          variants={itemVariants}
          className="mb-8 bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6"
        >
          <h2 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4 flex items-center gap-2">
            <Calendar className="h-5 w-5 text-indigo-500 dark:text-indigo-400" />
            Time Series Configuration
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Number of Months to Predict
              </label>
              <input
                type="number"
                min="1"
                step="1"
                value={noOfMonths}
                onChange={(e) => setNoOfMonths(e.target.value)}
                className="border rounded px-3 py-2 w-full text-gray-900 dark:text-gray-100 bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-600 focus:ring-indigo-500 dark:focus:ring-indigo-400 focus:border-indigo-500 dark:focus:border-indigo-400"
                disabled={isLoading}
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Date Column
              </label>
              <select
                value={dateColumn}
                onChange={(e) => setDateColumn(e.target.value)}
                className="border rounded px-3 py-2 w-full text-gray-900 dark:text-gray-100 bg-white dark:bg-gray-800 border-gray-300 dark:border-gray-600 focus:ring-indigo-500 dark:focus:ring-indigo-400 focus:border-indigo-500 dark:focus:border-indigo-400"
                disabled={isLoading}
              >
                <option value="">Select a date column</option>
                {columns.map((column, index) => (
                  <option key={index} value={column}>
                    {column}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </motion.div>

        {/* Status Messages */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="mb-6 p-4 rounded-lg bg-red-50 dark:bg-red-900/20 border-l-4 border-red-500 text-red-700 dark:text-red-400"
            >
              <div className="flex items-center">
                <Info className="h-5 w-5 mr-2 flex-shrink-0" />
                <p>{error}</p>
              </div>
            </motion.div>
          )}

          {success && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="mb-6 p-4 rounded-lg bg-green-50 dark:bg-green-900/20 border-l-4 border-green-500 text-green-700 dark:text-green-400"
            >
              <div className="flex items-center">
                <Check className="h-5 w-5 mr-2 flex-shrink-0" />
                <p>{success}</p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Action Button */}
        <motion.div variants={itemVariants} className="flex justify-end">
          <motion.button
            onClick={handleActionClick}
            disabled={isLoading}
            className={`flex items-center gap-2 px-6 py-2.5 rounded-lg font-medium transition-all ${
              isLoading
                ? "bg-indigo-400 dark:bg-indigo-600 text-white cursor-not-allowed"
                : "bg-indigo-600 text-white hover:bg-indigo-700 dark:bg-indigo-600 dark:hover:bg-indigo-700"
            }`}
            whileHover={isLoading ? {} : { scale: 1.05 }}
            whileTap={isLoading ? {} : { scale: 0.95 }}
          >
            {isLoading ? (
              <>
                <Loader2 className="h-5 w-5 animate-spin" />
                <span>Processing...</span>
              </>
            ) : (
              <>
                <TrendingUp className="h-5 w-5" />
                <span>Generate Projection</span>
              </>
            )}
          </motion.button>
        </motion.div>
      </div>

      {/* Plot Popup */}
      <AnimatePresence>
        {isPopupVisible && (
          <motion.div
            className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={closePopup}
          >
            <motion.div
              className="bg-white dark:bg-gray-800 rounded-xl shadow-xl w-full max-w-5xl mx-4 overflow-hidden"
              variants={popupVariants}
              initial="hidden"
              animate="visible"
              exit="hidden"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex justify-between items-center p-4 border-b border-gray-200 dark:border-gray-700">
                <h2 className="text-lg font-semibold text-gray-800 dark:text-gray-200 flex items-center gap-2">
                  <TrendingUp className="h-5 w-5 text-indigo-500 dark:text-indigo-400" />
                  Time Series Projection
                </h2>
                <div className="flex items-center gap-2">
                  <motion.button
                    onClick={() => {
                      navigate(`/projects/${project_id}/projection`, {
                        state: {
                          timeSeriesFigure: {
                            plotly_figure: JSON.stringify({
                              data: plotData,
                              layout: plotLayout,
                            }),
                          },
                        },
                      });
                    }}
                    className="px-3 py-1.5 text-sm font-medium text-indigo-600 dark:text-indigo-400 bg-indigo-50 dark:bg-indigo-900/30 rounded-md hover:bg-indigo-100 dark:hover:bg-indigo-900/50"
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    <Download className="h-4 w-4" />
                  </motion.button>
                  <motion.button
                    onClick={closePopup}
                    className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 rounded-full p-1"
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.9 }}
                  >
                    <X className="h-5 w-5" />
                  </motion.button>
                </div>
              </div>
              <div className="p-4">
                <Plot
                  data={plotData}
                  layout={{
                    ...plotLayout,
                    autosize: true,
                    width: undefined,
                    height: undefined,
                  }}
                  config={{ responsive: true }}
                  style={{ width: "100%", height: "100%", minHeight: "500px" }}
                />
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

export default Workbench;
