"use client";

import { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import {
  ChevronRight,
  ChevronLeft,
  Check,
  X,
  Filter,
  Columns,
  ArrowRight,
  Loader2,
  AlertCircle,
  Info,
} from "lucide-react";
import axios from "axios";
import Loader from "../Components/Loader";

const SelectColumns = () => {
  const { project_id, com_id } = useParams();
  const navigate = useNavigate();
  const [columns, setColumns] = useState([]);
  const [selectedColumns, setSelectedColumns] = useState([]);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [searchTerm, setSearchTerm] = useState("");
  const [showInfo, setShowInfo] = useState(true);
  const baseUrl = import.meta.env.VITE_BASE_URL;

  useEffect(() => {
    const fetchColumns = async () => {
      try {
        const response = await axios.get(
          `${baseUrl}/projects/${project_id}/columns`
        );
        setColumns(response.data.columns || []);
        // If there are already selected columns, use them
        if (
          response.data.selected_columns &&
          response.data.selected_columns.length > 0
        ) {
          setSelectedColumns(response.data.selected_columns);
        }
      } catch (err) {
        console.error("Error fetching columns:", err);
        setError("Failed to load columns. Please try again.");
      } finally {
        setLoading(false);
      }
    };

    fetchColumns();
  }, [project_id, baseUrl]);

  const handleSelectColumn = (column) => {
    if (selectedColumns.includes(column)) {
      setSelectedColumns(selectedColumns.filter((col) => col !== column));
    } else {
      setSelectedColumns([...selectedColumns, column]);
    }
  };

  const handleSelectAll = () => {
    if (selectedColumns.length === filteredColumns.length) {
      setSelectedColumns([]);
    } else {
      setSelectedColumns([...filteredColumns]);
    }
  };

  const handleSubmit = async () => {
    if (selectedColumns.length === 0) {
      setError("Please select at least one column");
      return;
    }

    setSubmitting(true);
    setError("");

    try {
      await axios.post(`${baseUrl}/projects/${project_id}/columns`, {
        selected_columns: selectedColumns,
      });

      // Navigate to the next step
      navigate(`/projects/${project_id}/select-kpi`);
    } catch (err) {
      console.error("Error submitting columns:", err);
      setError("Failed to save selected columns. Please try again.");
    } finally {
      setSubmitting(false);
    }
  };

  const filteredColumns = columns.filter((column) =>
    column.toLowerCase().includes(searchTerm.toLowerCase())
  );

  if (loading) {
    return <Loader />;
  }

  // Animation variants
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        when: "beforeChildren",
        staggerChildren: 0.05,
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

  return (
    <div className="container mx-auto px-4 py-8 max-w-6xl">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
            Select Columns
          </h1>
          <p className="text-gray-600 dark:text-gray-400">
            Choose the columns you want to include in your analysis.
          </p>
        </div>

        {/* Info Card */}
        {showInfo && (
          <motion.div
            className="mb-6 bg-blue-50 dark:bg-blue-900/20 border-l-4 border-blue-500 p-4 rounded-lg"
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <div className="flex">
              <div className="flex-shrink-0">
                <Info className="h-5 w-5 text-blue-500" />
              </div>
              <div className="ml-3">
                <h3 className="text-sm font-medium text-blue-800 dark:text-blue-300">
                  Column Selection Tips
                </h3>
                <div className="mt-2 text-sm text-blue-700 dark:text-blue-200">
                  <ul className="list-disc pl-5 space-y-1">
                    <li>Select columns that are relevant to your analysis</li>
                    <li>Exclude columns with too many missing values</li>
                    <li>
                      Include both numerical and categorical features for better
                      insights
                    </li>
                  </ul>
                </div>
              </div>
              <div className="ml-auto pl-3">
                <div className="-mx-1.5 -my-1.5">
                  <button
                    onClick={() => setShowInfo(false)}
                    className="inline-flex rounded-md p-1.5 text-blue-500 hover:bg-blue-100 dark:hover:bg-blue-800 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                  >
                    <span className="sr-only">Dismiss</span>
                    <X className="h-5 w-5" />
                  </button>
                </div>
              </div>
            </div>
          </motion.div>
        )}

        {/* Error Message */}
        {error && (
          <motion.div
            className="mb-6 p-4 rounded-lg bg-red-50 dark:bg-red-900/20 border-l-4 border-red-500 flex items-start"
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <AlertCircle className="h-5 w-5 text-red-500 mr-3 mt-0.5" />
            <p className="text-sm text-red-700 dark:text-red-300">{error}</p>
          </motion.div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Available Columns */}
          <div className="lg:col-span-1 bg-white dark:bg-gray-800 rounded-xl shadow-md overflow-hidden">
            <div className="p-4 bg-gray-50 dark:bg-gray-700 border-b border-gray-200 dark:border-gray-600 flex justify-between items-center">
              <div className="flex items-center">
                <Columns className="h-5 w-5 text-gray-500 dark:text-gray-400 mr-2" />
                <h2 className="text-lg font-medium text-gray-900 dark:text-white">
                  Available Columns
                </h2>
              </div>
              <span className="text-sm text-gray-500 dark:text-gray-400 bg-gray-200 dark:bg-gray-600 px-2 py-1 rounded-full">
                {filteredColumns.length}
              </span>
            </div>

            {/* Search Box */}
            <div className="p-4 border-b border-gray-200 dark:border-gray-600">
              <div className="relative">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Filter className="h-4 w-4 text-gray-400" />
                </div>
                <input
                  type="text"
                  className="block w-full pl-10 pr-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:text-white sm:text-sm"
                  placeholder="Search columns..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                />
              </div>
            </div>

            {/* Select All Button */}
            <div className="p-3 border-b border-gray-200 dark:border-gray-600">
              <button
                onClick={handleSelectAll}
                className="w-full flex items-center justify-center px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm text-sm font-medium text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-700 hover:bg-gray-50 dark:hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
              >
                {selectedColumns.length === filteredColumns.length ? (
                  <>
                    <X className="h-4 w-4 mr-2" />
                    Deselect All
                  </>
                ) : (
                  <>
                    <Check className="h-4 w-4 mr-2" />
                    Select All
                  </>
                )}
              </button>
            </div>

            {/* Column List */}
            <div className="p-2 max-h-[400px] overflow-y-auto">
              <motion.ul
                className="space-y-1"
                variants={containerVariants}
                initial="hidden"
                animate="visible"
              >
                {filteredColumns.length > 0 ? (
                  filteredColumns.map((column) => (
                    <motion.li key={column} variants={itemVariants}>
                      <button
                        onClick={() => handleSelectColumn(column)}
                        className={`w-full flex items-center px-3 py-2 text-sm rounded-md transition-colors ${
                          selectedColumns.includes(column)
                            ? "bg-indigo-100 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-300"
                            : "text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"
                        }`}
                      >
                        <div
                          className={`flex-shrink-0 h-4 w-4 mr-3 rounded border ${
                            selectedColumns.includes(column)
                              ? "bg-indigo-600 border-indigo-600 flex items-center justify-center"
                              : "border-gray-300 dark:border-gray-600"
                          }`}
                        >
                          {selectedColumns.includes(column) && (
                            <Check className="h-3 w-3 text-white" />
                          )}
                        </div>
                        <span className="truncate">{column}</span>
                      </button>
                    </motion.li>
                  ))
                ) : (
                  <li className="px-3 py-6 text-center text-sm text-gray-500 dark:text-gray-400">
                    No columns found matching your search
                  </li>
                )}
              </motion.ul>
            </div>
          </div>

          {/* Middle Section with Arrows */}
          <div className="hidden lg:flex lg:col-span-1 items-center justify-center">
            <div className="flex flex-col items-center space-y-4">
              <motion.div
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                className="p-3 rounded-full bg-indigo-100 dark:bg-indigo-900/30"
              >
                <ChevronRight className="h-6 w-6 text-indigo-600 dark:text-indigo-400" />
              </motion.div>
              <p className="text-sm text-gray-500 dark:text-gray-400 text-center">
                Select columns to include in your analysis
              </p>
              <motion.div
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                className="p-3 rounded-full bg-indigo-100 dark:bg-indigo-900/30"
              >
                <ChevronLeft className="h-6 w-6 text-indigo-600 dark:text-indigo-400" />
              </motion.div>
            </div>
          </div>

          {/* Selected Columns */}
          <div className="lg:col-span-1 bg-white dark:bg-gray-800 rounded-xl shadow-md overflow-hidden">
            <div className="p-4 bg-gray-50 dark:bg-gray-700 border-b border-gray-200 dark:border-gray-600 flex justify-between items-center">
              <div className="flex items-center">
                <Check className="h-5 w-5 text-green-500 mr-2" />
                <h2 className="text-lg font-medium text-gray-900 dark:text-white">
                  Selected Columns
                </h2>
              </div>
              <span className="text-sm text-gray-500 dark:text-gray-400 bg-gray-200 dark:bg-gray-600 px-2 py-1 rounded-full">
                {selectedColumns.length}
              </span>
            </div>

            {/* Selected Column List */}
            <div className="p-2 h-[500px] overflow-y-auto">
              {selectedColumns.length > 0 ? (
                <motion.ul
                  className="space-y-1"
                  variants={containerVariants}
                  initial="hidden"
                  animate="visible"
                >
                  {selectedColumns.map((column) => (
                    <motion.li key={column} variants={itemVariants}>
                      <div className="flex items-center justify-between px-3 py-2 text-sm rounded-md bg-indigo-50 dark:bg-indigo-900/20 text-indigo-700 dark:text-indigo-300">
                        <span className="truncate">{column}</span>
                        <button
                          onClick={() => handleSelectColumn(column)}
                          className="ml-2 p-1 rounded-full hover:bg-indigo-100 dark:hover:bg-indigo-800"
                        >
                          <X className="h-4 w-4 text-indigo-600 dark:text-indigo-400" />
                        </button>
                      </div>
                    </motion.li>
                  ))}
                </motion.ul>
              ) : (
                <div className="flex flex-col items-center justify-center h-full text-center p-6">
                  <div className="p-4 bg-gray-100 dark:bg-gray-700 rounded-full mb-4">
                    <Columns className="h-8 w-8 text-gray-400" />
                  </div>
                  <h3 className="text-base font-medium text-gray-900 dark:text-white mb-1">
                    No columns selected
                  </h3>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    Select columns from the left panel to include them in your
                    analysis
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Navigation Buttons */}
        <div className="mt-8 flex justify-between">
          <motion.button
            onClick={() => navigate(`/${com_id}/projects/${project_id}/upload`)}
            className="inline-flex items-center px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm text-sm font-medium text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <ChevronLeft className="-ml-1 mr-2 h-5 w-5" />
            Back to Upload
          </motion.button>

          <motion.button
            onClick={handleSubmit}
            disabled={selectedColumns.length === 0 || submitting}
            className="inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed"
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            {submitting ? (
              <>
                <Loader2 className="animate-spin -ml-1 mr-2 h-5 w-5" />
                Saving...
              </>
            ) : (
              <>
                Continue to Select KPI
                <ArrowRight className="ml-2 h-5 w-5" />
              </>
            )}
          </motion.button>
        </div>
      </motion.div>
    </div>
  );
};

export default SelectColumns;
