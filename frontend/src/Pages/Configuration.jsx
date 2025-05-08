"use client";

import React, { useState, useEffect, useRef } from "react";
import PropTypes from "prop-types";
import { useNavigate, useParams } from "react-router-dom";
import { motion } from "framer-motion";
import {
  Upload,
  Database,
  FileText,
  ChevronRight,
  AlertCircle,
  CheckCircle,
  X,
  Loader2,
  Settings,
  ChevronLeft,
} from "lucide-react";
import { useSelector, useDispatch } from "react-redux";
import { updateProject, createProject } from "../redux/projectsSlice";
import Select from "react-select";
import axiosInstance from "../utils/axiosInstance";
import { getAuthState } from "../utils/auth";
import ChatBot from "../components/ChatBot";
import { FaRobot } from "react-icons/fa";

// Add DUMMY_COLUMNS definition at the top of the file
const DUMMY_COLUMNS = [
  {
    id: 1,
    name: "customer_id",
    type: "string",
    description: "Unique identifier for each customer",
  },
  {
    id: 2,
    name: "transaction_date",
    type: "date",
    description: "Date of transaction",
  },
  {
    id: 3,
    name: "amount",
    type: "numeric",
    description: "Transaction amount",
  },
  {
    id: 4,
    name: "product_category",
    type: "categorical",
    description: "Category of product",
  },
];

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex items-center justify-center h-screen">
          <div className="text-center">
            <h2 className="text-2xl font-bold text-red-600 mb-4">
              Something went wrong
            </h2>
            <p className="text-gray-600">{this.state.error?.message}</p>
            <button
              onClick={() => this.setState({ hasError: false, error: null })}
              className="mt-4 px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600"
            >
              Try again
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

ErrorBoundary.propTypes = {
  children: PropTypes.node.isRequired,
};

function ConfigurationContent() {
  const [activeTab, setActiveTab] = useState("upload");
  const navigate = useNavigate();
  const dispatch = useDispatch();
  const { project_id } = useParams();
  const { com_id } = getAuthState();
  const [reUpload, setReUpload] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  // Upload states
  const [file, setFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const [uploadSuccess, setUploadSuccess] = useState("");
  const [loading, setLoading] = useState(false);
  const fileInputRef = useRef(null);
  const [validated, setValidated] = useState(false);

  // Column selection states
  const [columns, setColumns] = useState(DUMMY_COLUMNS);
  const [droppedColumns, setDroppedColumns] = useState([]);
  const [selectedKpi, setSelectedKpi] = useState([]);
  const [selectedImportant, setSelectedImportant] = useState([]);
  const [columnError, setColumnError] = useState("");
  const [columnSuccess, setColumnSuccess] = useState("");
  const [droppingColumns, setDroppingColumns] = useState(false);
  const [uploadedFileData, setUploadedFileData] = useState([]);
  const [importantColumnNames, setImportantColumnNames] = useState([]);

  // Get project from Redux store
  const project = useSelector((state) =>
    state.projects.find(
      (project) =>
        project.projectId === project_id || project.project_id === project_id
    )
  );

  const storedDroppedColumns = project?.droppedColumns;
  const kpiList = project?.kpiList;

  // Add toast state
  const [showSuccessToast, setShowSuccessToast] = useState(false);

  // Combined useEffect for initialization and reUpload check
  useEffect(() => {
    try {
      console.log("Configuration component mounted");
      console.log("Company ID:", com_id);
      console.log("Project ID:", project_id);
      console.log("Project data:", project);

      if (!com_id) {
        setError("Company ID is not set. Please log in again.");
        return;
      }

      // Set loading state
      if (!project && project_id) {
        setIsLoading(true);
        // Try to fetch project data if not in store
        const fetchProject = async () => {
          try {
            const response = await axiosInstance.get(
              `/${com_id}/projects/${project_id}`
            );
            if (response.data) {
              dispatch(
                createProject({
                  projectId: project_id,
                  project_id: project_id,
                  columns: response.data.columns || [],
                  importantColumnNames:
                    response.data.importantColumnNames || [],
                  kpiList: response.data.kpiList || [],
                  droppedColumns: response.data.droppedColumns || [],
                  uploadedFileData: response.data.uploadedFileData || [],
                  selectedKpi: response.data.selectedKpi || null,
                })
              );
            }
          } catch (error) {
            console.error("Error fetching project:", error);
            setError(error);
          } finally {
            setIsLoading(false);
          }
        };
        fetchProject();
      } else {
        setIsLoading(false);
      }

      // Check for reUpload condition
      if (
        uploadedFileData?.length > 0 &&
        storedDroppedColumns?.length > 0 &&
        importantColumnNames?.length > 0 &&
        kpiList?.length > 0
      ) {
        setReUpload(true);
      }
    } catch (err) {
      console.error("Error in Configuration component:", err);
      setError(err);
    }
  }, [
    com_id,
    project_id,
    project,
    uploadedFileData,
    storedDroppedColumns,
    importantColumnNames,
    kpiList,
    dispatch,
  ]);

  const [isChatOpen, setIsChatOpen] = useState(false);

  if (error) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center">
          <h2 className="text-2xl font-bold text-red-600 mb-4">
            Something went wrong
          </h2>
          <p className="text-gray-600">{error.message}</p>
          <button
            onClick={() => setError(null)}
            className="mt-4 px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600"
          >
            Try again
          </button>
        </div>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mb-4"></div>
          <p className="text-gray-600">Loading project data...</p>
        </div>
      </div>
    );
  }

  // Handle file upload functions
  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const droppedFile = e.dataTransfer.files[0];
      if (validateFile(droppedFile)) {
        setFile(droppedFile);
        setUploadError("");
      }
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      const selectedFile = e.target.files[0];
      if (validateFile(selectedFile)) {
        setFile(selectedFile);
        setUploadError("");
      }
    }
  };

  const validateFile = (file) => {
    // Check file type (CSV or Excel)
    const validTypes = [
      "text/csv",
      "application/vnd.ms-excel",
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ];
    if (!validTypes.includes(file.type)) {
      setUploadError("Please upload a CSV or Excel file");
      return false;
    }

    // Check file size (max 10MB)
    const maxSize = 10 * 1024 * 1024; // 10MB
    if (file.size > maxSize) {
      setUploadError("File size should not exceed 10MB");
      return false;
    }

    return true;
  };

  const handleRemoveFile = () => {
    console.log("handleRemoveFile called");
    setFile(null);
    setUploadSuccess("");
    setValidated(false);
    setUploadError("");
    setColumns(DUMMY_COLUMNS);
    setImportantColumnNames([]);
    setDroppedColumns([]);
    setUploadedFileData([]);
    setIsChatOpen(false);
    localStorage.removeItem(`chat_messages_${project_id}`);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleFileUpload = async (e) => {
    e.preventDefault();
    try {
      setLoading(true);
      setError(null);

      if (!file) {
        throw new Error("Please select a file to upload");
      }

      const formData = new FormData();
      formData.append("file", file);

      const response = await axiosInstance.post(
        `/ingest/${project_id}`,
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      if (response.data && response.data.columns) {
        // Format columns for dropdown
        const formattedColumns = response.data.columns.map((name, index) => ({
          id: index + 1,
          name: name,
          type: "string", // Default type
          description: name, // Use column name as description
        }));

        setColumns(formattedColumns);
        setImportantColumnNames(response.data.importantColumnNames || []);
        setDroppedColumns(response.data.droppedColumns || []);
        setUploadedFileData(response.data.uploadedFileData || []);
      }

      // Show success message
      setUploadSuccess("File uploaded and validated successfully!");
      setValidated(true);
      setShowSuccessToast(true); // Show toast
      setTimeout(() => setShowSuccessToast(false), 3000); // Hide after 3s

      // Reset chatbot state
      setIsChatOpen(false);
      localStorage.removeItem(`chat_messages_${project_id}`);
    } catch (error) {
      console.error("Error uploading file:", error);
      setError(
        error.response?.data?.message ||
          error.message ||
          "Failed to upload file"
      );
    } finally {
      setLoading(false);
    }
  };

  const handleDropColumns = async () => {
    if (droppedColumns.length === 0) {
      setColumnError("Please select at least one column to drop");
      return;
    }

    setDroppingColumns(true);
    setColumnError("");
    setColumnSuccess("");

    try {
      const columnNames = droppedColumns
        .map((colId) => {
          const column = columns.find((c) => c.id === colId);
          return column ? column.name : null;
        })
        .filter((name) => name !== null);

      // Check if any of the columns are already dropped
      const alreadyDropped = columnNames.filter((name) =>
        storedDroppedColumns?.includes(name)
      );

      if (alreadyDropped.length > 0) {
        setColumnSuccess(
          `Columns ${alreadyDropped.join(", ")} were already dropped.`
        );
        // Remove already dropped columns from the list
        const newColumnsToDrop = columnNames.filter(
          (name) => !alreadyDropped.includes(name)
        );

        if (newColumnsToDrop.length === 0) {
          setDroppingColumns(false);
          return;
        }

        // Only drop the new columns
        const response = await axiosInstance.post(
          `/projects/${project_id}/drop_columns`,
          {
            project_id: project_id,
            columns: newColumnsToDrop,
          }
        );

        if (response.status === 200) {
          setColumnSuccess(
            `Successfully dropped columns: ${newColumnsToDrop.join(", ")}`
          );
          setColumns(
            columns.filter((col) => !newColumnsToDrop.includes(col.name))
          );
          dispatch(
            updateProject({
              projectId: project_id,
              droppedColumns: [...storedDroppedColumns, ...newColumnsToDrop],
              columns: columns.filter(
                (col) => !newColumnsToDrop.includes(col.name)
              ),
            })
          );
          setDroppedColumns([]);
        }
      } else {
        // No columns were already dropped, proceed with normal drop
        const response = await axiosInstance.post(
          `/projects/${project_id}/drop_columns`,
          {
            project_id: project_id,
            columns: columnNames,
          }
        );

        if (response.status === 200) {
          setColumnSuccess("Columns dropped successfully!");
          setColumns(columns.filter((col) => !columnNames.includes(col.name)));
          dispatch(
            updateProject({
              projectId: project_id,
              droppedColumns: columnNames,
              columns: columns.filter((col) => !columnNames.includes(col.name)),
            })
          );
          setDroppedColumns([]);
        } else {
          setColumnError("Failed to drop columns: " + response.data.message);
        }
      }
    } catch (error) {
      console.error("Error dropping columns:", error);
      if (
        error.response?.status === 500 &&
        error.response?.data?.detail === "Project not found or no change made."
      ) {
        setColumnSuccess("Columns were already dropped.");
        setDroppedColumns([]);
      } else {
        setColumnError(
          error.response?.data?.message ||
            "Failed to drop columns. Please try again."
        );
      }
    } finally {
      setDroppingColumns(false);
    }
  };

  const handleSubmitColumns = async () => {
    if (selectedKpi.length === 0 || selectedImportant.length === 0) {
      setColumnError(
        "Please select both KPI and Important columns before submitting."
      );
      return;
    }

    const importantColumns = selectedImportant.map((colId) =>
      columns.find((col) => col.id === colId)
    );
    const kpiColumns = selectedKpi.map((colId) =>
      columns.find((col) => col.id === colId)
    );

    try {
      // First update the project with the selected columns
      dispatch(
        updateProject({
          projectId: project_id,
          importantColumnNames: importantColumns.map((col) => col.name),
          kpiList: kpiColumns.map((col) => col.name),
          selectedKpi: kpiColumns,
          selectedImportant: importantColumns,
        })
      );

      // Check if KPI columns are already set
      if (project?.kpiList?.length > 0) {
        console.log("KPI columns already set, proceeding with navigation");
      } else {
        // Set KPI columns in the backend only if they haven't been set yet
        await axiosInstance.post(`/projects/${project_id}/kpi_columns`, {
          columns: kpiColumns.map((col) => col.name),
        });
      }

      // Set important columns in the backend only if they haven't been set yet
      if (!project?.importantColumnNames?.length) {
        await axiosInstance.post(`/projects/${project_id}/important_columns`, {
          columns: importantColumns.map((col) => col.name),
        });
      }

      // Navigate to the next step
      navigate(`/${com_id}/projects/${project_id}/select-kpi`, {
        state: {
          importantColumns,
          kpiColumns,
        },
      });
    } catch (error) {
      console.error("Error submitting columns:", error);
      if (
        error.response?.status === 500 &&
        error.response?.data?.detail === "Project not found or no change made."
      ) {
        // If KPI columns are already set, proceed with navigation
        if (project?.kpiList?.length > 0) {
          navigate(`/${com_id}/projects/${project_id}/select-kpi`, {
            state: {
              importantColumns,
              kpiColumns,
            },
          });
        } else {
          setColumnError(
            "Failed to submit column selections. Please try again."
          );
        }
      } else {
        setColumnError(
          error.response?.data?.message ||
            "Failed to submit column selections. Please try again."
        );
      }
    }
  };

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

  return (
    <div className="bg-gray-50 dark:bg-gray-900 min-h-[calc(100vh-64px)]">
      <div className="container mx-auto px-4 py-8">
        <motion.div
          className="mb-8"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
            <div className="flex items-center gap-3">
              <motion.button
                onClick={() => navigate(`/${com_id}`)}
                className="p-2 rounded-full bg-white dark:bg-gray-800 shadow-sm border border-gray-200 dark:border-gray-700"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <ChevronLeft className="h-5 w-5 text-gray-600 dark:text-gray-400" />
              </motion.button>
              <div>
                <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
                  Project Configuration
                </h1>
                <p className="text-gray-600 dark:text-gray-400 mt-1">
                  Configure your project settings and data
                </p>
              </div>
            </div>
          </div>
        </motion.div>

        {reUpload ? (
          <motion.div
            variants={containerVariants}
            initial="hidden"
            animate="visible"
            className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden"
          >
            <div className="p-6">
              <motion.h2
                variants={itemVariants}
                className="text-xl font-semibold text-gray-900 dark:text-white mb-6"
              >
                Project Configuration
              </motion.h2>

              <motion.div
                variants={itemVariants}
                className="grid md:grid-cols-2 gap-6"
              >
                <div className="space-y-4">
                  <div className="bg-gray-50 dark:bg-gray-900/50 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
                    <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-3">
                      Uploaded Files
                    </h3>
                    <div className="flex flex-wrap gap-2">
                      {uploadedFileData && (
                        <div className="flex items-center gap-2 px-3 py-1.5 bg-indigo-50 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-300 rounded-md text-sm">
                          <FileText className="h-4 w-4" />
                          <span>{uploadedFileData}</span>
                        </div>
                      )}
                    </div>
                  </div>

                  <div className="bg-gray-50 dark:bg-gray-900/50 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
                    <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-3">
                      Dropped Columns
                    </h3>
                    <div className="flex flex-wrap gap-2">
                      {storedDroppedColumns?.map((col, index) => (
                        <div
                          key={index}
                          className="px-3 py-1.5 bg-red-50 dark:bg-red-900/30 text-red-700 dark:text-red-300 rounded-md text-sm"
                        >
                          {col}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                <div className="space-y-4">
                  <div className="bg-gray-50 dark:bg-gray-900/50 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
                    <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-3">
                      Important Columns
                    </h3>
                    <div className="flex flex-wrap gap-2">
                      {importantColumnNames?.map((col, index) => (
                        <div
                          key={index}
                          className="px-3 py-1.5 bg-green-50 dark:bg-green-900/30 text-green-700 dark:text-green-300 rounded-md text-sm"
                        >
                          {col}
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="bg-gray-50 dark:bg-gray-900/50 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
                    <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-3">
                      KPI Columns
                    </h3>
                    <div className="flex flex-wrap gap-2">
                      {kpiList?.map((kpi, index) => (
                        <div
                          key={index}
                          className="px-3 py-1.5 bg-purple-50 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 rounded-md text-sm"
                        >
                          {kpi}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </motion.div>

              <motion.div
                variants={itemVariants}
                className="mt-8 flex flex-col sm:flex-row gap-4"
              >
                <motion.button
                  className="btn btn-primary flex items-center gap-2"
                  onClick={() => {
                    navigate(`/projects/${project_id}/clustered-data`, {
                      state: { project_id, importantColumnNames, kpiList },
                    });
                  }}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <Database className="h-4 w-4" />
                  <span>Continue Analysis</span>
                </motion.button>

                <motion.button
                  className="btn btn-outline flex items-center gap-2"
                  onClick={() => {
                    localStorage.setItem("uploadedFileData", "");
                    setReUpload(false);
                    setActiveTab("upload");
                  }}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <Upload className="h-4 w-4" />
                  <span>Upload New Data</span>
                </motion.button>
              </motion.div>
            </div>
          </motion.div>
        ) : (
          <motion.div
            variants={containerVariants}
            initial="hidden"
            animate="visible"
            className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden"
          >
            <div className="border-b border-gray-200 dark:border-gray-700">
              <nav className="flex">
                <button
                  onClick={() => setActiveTab("upload")}
                  className={`px-6 py-4 text-sm font-medium flex items-center gap-2 ${
                    activeTab === "upload"
                      ? "border-b-2 border-indigo-500 text-indigo-600 dark:text-indigo-400 dark:border-indigo-400"
                      : "border-b-2 border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 dark:text-gray-400 dark:hover:text-gray-300"
                  }`}
                >
                  <Upload className="h-4 w-4" />
                  Upload Data
                </button>
                <button
                  onClick={() => setActiveTab("settings")}
                  disabled={!validated && !reUpload}
                  className={`px-6 py-4 text-sm font-medium flex items-center gap-2 ${
                    activeTab === "settings"
                      ? "border-b-2 border-indigo-500 text-indigo-600 dark:text-indigo-400 dark:border-indigo-400"
                      : "border-b-2 border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 dark:text-gray-400 dark:hover:text-gray-300"
                  } ${
                    !validated && !reUpload
                      ? "opacity-50 cursor-not-allowed"
                      : ""
                  }`}
                >
                  <Settings className="h-4 w-4" />
                  Settings
                </button>
              </nav>
            </div>

            <div className="p-6">
              {activeTab === "upload" && (
                <motion.div variants={itemVariants} className="space-y-6">
                  <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
                    Upload Your Data
                  </h2>
                  <p className="text-gray-600 dark:text-gray-400">
                    Upload a CSV or Excel file to start analyzing your data.
                    Maximum file size is 10MB.
                  </p>

                  {/* Error Message */}
                  {uploadError && (
                    <motion.div
                      className="mb-6 p-4 rounded-lg bg-red-50 dark:bg-red-900/20 border-l-4 border-red-500 flex items-start"
                      initial={{ opacity: 0, y: -10 }}
                      animate={{ opacity: 1, y: 0 }}
                    >
                      <AlertCircle className="h-5 w-5 text-red-500 mr-3 mt-0.5" />
                      <div>
                        <h3 className="text-sm font-medium text-red-800 dark:text-red-400">
                          Error
                        </h3>
                        <p className="text-sm text-red-700 dark:text-red-300 mt-1">
                          {uploadError}
                        </p>
                      </div>
                    </motion.div>
                  )}

                  {/* Success Message (Toast) */}
                  {showSuccessToast && (
                    <div className="fixed top-6 right-6 z-50">
                      <div className="flex items-center p-4 bg-green-500 text-white rounded-lg shadow-lg animate-fade-in-up">
                        <CheckCircle className="h-6 w-6 mr-2" />
                        <span>File uploaded and validated successfully!</span>
                      </div>
                    </div>
                  )}

                  {/* Drag & Drop Area */}
                  {!file ? (
                    <div
                      className={`border-2 border-dashed rounded-lg p-12 text-center cursor-pointer transition-colors ${
                        isDragging
                          ? "border-indigo-500 bg-indigo-50 dark:bg-indigo-900/20"
                          : "border-gray-300 dark:border-gray-600 hover:border-indigo-400 hover:bg-gray-50 dark:hover:bg-gray-700/50"
                      }`}
                      onDragOver={handleDragOver}
                      onDragLeave={handleDragLeave}
                      onDrop={handleDrop}
                      onClick={() => fileInputRef.current?.click()}
                    >
                      <input
                        type="file"
                        ref={fileInputRef}
                        onChange={handleFileChange}
                        accept=".csv,.xls,.xlsx"
                        className="hidden"
                      />
                      <div className="flex flex-col items-center justify-center">
                        <div className="mb-4 p-4 bg-indigo-100 dark:bg-indigo-900/30 rounded-full">
                          <Upload className="h-8 w-8 text-indigo-600 dark:text-indigo-400" />
                        </div>
                        <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                          {isDragging
                            ? "Drop your file here"
                            : "Drag & Drop your file here"}
                        </h3>
                        <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
                          or click to browse from your computer
                        </p>
                        <p className="text-xs text-gray-400 dark:text-gray-500">
                          Supported formats: CSV, Excel (.xls, .xlsx)
                        </p>
                      </div>
                    </div>
                  ) : (
                    <div className="border rounded-lg p-6">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center">
                          <div className="p-3 bg-indigo-100 dark:bg-indigo-900/30 rounded-lg mr-4">
                            <FileText className="h-6 w-6 text-indigo-600 dark:text-indigo-400" />
                          </div>
                          <div>
                            <h3 className="text-base font-medium text-gray-900 dark:text-white">
                              {file.name}
                            </h3>
                            <p className="text-sm text-gray-500 dark:text-gray-400">
                              {(file.size / 1024 / 1024).toFixed(2)} MB
                            </p>
                          </div>
                        </div>
                        <button
                          onClick={handleRemoveFile}
                          className="p-1.5 rounded-full bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
                          aria-label="Remove file"
                          disabled={loading}
                        >
                          <X className="h-4 w-4 text-gray-500 dark:text-gray-400" />
                        </button>
                      </div>
                    </div>
                  )}

                  {/* Validation Status */}
                  {loading && (
                    <motion.div
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg flex items-center"
                    >
                      <Loader2 className="h-5 w-5 text-blue-500 animate-spin mr-3" />
                      <p className="text-sm text-blue-700 dark:text-blue-300">
                        Validating file, please wait...
                      </p>
                    </motion.div>
                  )}

                  {/* Buttons */}
                  <div className="flex justify-between items-center mt-4">
                    <motion.button
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      onClick={handleFileUpload}
                      disabled={!file || loading || validated}
                      className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50"
                    >
                      {loading ? (
                        <div className="flex items-center space-x-2">
                          <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                          <span>Validating...</span>
                        </div>
                      ) : (
                        "Upload & Validate"
                      )}
                    </motion.button>
                    <div className="flex space-x-4">
                      <motion.button
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        onClick={() => setIsChatOpen(true)}
                        disabled={!validated}
                        className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-colors ${
                          validated
                            ? "bg-blue-600 text-white hover:bg-blue-700"
                            : "bg-gray-300 text-gray-500 cursor-not-allowed"
                        }`}
                      >
                        <FaRobot className="text-xl" />
                        <span>AI Assistant</span>
                      </motion.button>
                      <motion.button
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        onClick={() => setActiveTab("settings")}
                        disabled={!validated}
                        className={`px-6 py-2 rounded-lg transition-colors ${
                          validated
                            ? "bg-green-600 text-white hover:bg-green-700"
                            : "bg-gray-300 text-gray-500 cursor-not-allowed"
                        }`}
                      >
                        Go to Settings
                      </motion.button>
                    </div>
                  </div>

                  {/* Tips Section */}
                  <div className="mt-8 bg-gray-50 dark:bg-gray-700/50 p-6 border-t border-gray-200 dark:border-gray-700 rounded-lg">
                    <h4 className="text-sm font-medium text-gray-900 dark:text-white mb-3">
                      Tips for successful upload:
                    </h4>
                    <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-300">
                      <li className="flex items-start">
                        <span className="inline-flex items-center justify-center h-5 w-5 rounded-full bg-indigo-100 dark:bg-indigo-900/30 text-indigo-600 dark:text-indigo-400 mr-2 text-xs">
                          1
                        </span>
                        Ensure your data is clean and properly formatted
                      </li>
                      <li className="flex items-start">
                        <span className="inline-flex items-center justify-center h-5 w-5 rounded-full bg-indigo-100 dark:bg-indigo-900/30 text-indigo-600 dark:text-indigo-400 mr-2 text-xs">
                          2
                        </span>
                        Include headers in your CSV/Excel file
                      </li>
                      <li className="flex items-start">
                        <span className="inline-flex items-center justify-center h-5 w-5 rounded-full bg-indigo-100 dark:bg-indigo-900/30 text-indigo-600 dark:text-indigo-400 mr-2 text-xs">
                          3
                        </span>
                        Remove any sensitive or personally identifiable
                        information
                      </li>
                    </ul>
                  </div>
                </motion.div>
              )}

              {activeTab === "settings" && (
                <motion.div
                  variants={containerVariants}
                  initial="hidden"
                  animate="visible"
                  className="space-y-6"
                >
                  <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm p-6">
                    <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-6">
                      Column Configuration
                    </h2>

                    {/* Drop Columns Section */}
                    <div className="mb-8 space-y-3">
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                        Select Columns to Drop
                      </label>
                      <div className="relative">
                        <Select
                          isMulti
                          menuCloseOnSelect={false}
                          closeMenuOnSelect={false}
                          options={columns.map((col) => ({
                            value: col.id,
                            label: col.name,
                          }))}
                          value={droppedColumns.map((id) => ({
                            value: id,
                            label: columns.find((col) => col.id === id)?.name,
                          }))}
                          onChange={(selected) => {
                            setDroppedColumns(
                              selected.map((option) => option.value)
                            );
                          }}
                          placeholder="Select columns to drop..."
                          className="react-select-container"
                          classNamePrefix="react-select"
                          styles={{
                            control: (base) => ({
                              ...base,
                              borderRadius: "0.5rem",
                              borderColor: "rgb(209 213 219)",
                              boxShadow: "none",
                              "&:hover": {
                                borderColor: "rgb(79 70 229)",
                              },
                              padding: "2px",
                              backgroundColor: "transparent",
                            }),
                            multiValue: (base) => ({
                              ...base,
                              backgroundColor: "rgb(238 242 255)",
                              borderRadius: "0.375rem",
                            }),
                            multiValueLabel: (base) => ({
                              ...base,
                              color: "rgb(67 56 202)",
                              fontWeight: 500,
                            }),
                            multiValueRemove: (base) => ({
                              ...base,
                              color: "rgb(67 56 202)",
                              ":hover": {
                                backgroundColor: "rgb(224 231 255)",
                                color: "rgb(79 70 229)",
                              },
                            }),
                            menu: (base) => ({
                              ...base,
                              borderRadius: "0.5rem",
                              overflow: "hidden",
                              boxShadow:
                                "0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)",
                            }),
                            option: (base, state) => ({
                              ...base,
                              backgroundColor: state.isSelected
                                ? "rgb(79 70 229)"
                                : state.isFocused
                                ? "rgb(238 242 255)"
                                : base.backgroundColor,
                              color: state.isSelected ? "white" : "inherit",
                              ":active": {
                                backgroundColor: state.isSelected
                                  ? "rgb(67 56 202)"
                                  : "rgb(224 231 255)",
                              },
                            }),
                          }}
                          theme={(theme) => ({
                            ...theme,
                            colors: {
                              ...theme.colors,
                              primary: "rgb(79 70 229)",
                              primary75: "rgb(99 102 241)",
                              primary50: "rgb(129 140 248)",
                              primary25: "rgb(224 231 255)",
                            },
                          })}
                        />
                      </div>
                      <motion.button
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                        onClick={handleDropColumns}
                        disabled={
                          droppedColumns.length === 0 || droppingColumns
                        }
                        className={`mt-2 px-4 py-2.5 rounded-md text-white flex items-center gap-2 transition-all ${
                          droppedColumns.length === 0 || droppingColumns
                            ? "bg-gray-400 cursor-not-allowed opacity-70"
                            : "bg-red-500 hover:bg-red-600 shadow-sm hover:shadow"
                        }`}
                      >
                        {droppingColumns ? (
                          <>
                            <Loader2 className="animate-spin h-4 w-4" />
                            <span>Dropping Columns...</span>
                          </>
                        ) : (
                          <>
                            <X className="h-4 w-4" />
                            <span>Drop Selected Columns</span>
                          </>
                        )}
                      </motion.button>
                    </div>

                    {/* KPI Columns Section */}
                    <div className="mb-8 space-y-3">
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                        Select KPI Columns
                      </label>
                      <div className="relative">
                        <Select
                          isMulti
                          menuCloseOnSelect={false}
                          closeMenuOnSelect={false}
                          options={columns
                            .filter((col) => !droppedColumns.includes(col.id))
                            .map((col) => ({ value: col.id, label: col.name }))}
                          value={selectedKpi.map((id) => ({
                            value: id,
                            label: columns.find((col) => col.id === id)?.name,
                          }))}
                          onChange={(selected) => {
                            setSelectedKpi(
                              selected.map((option) => option.value)
                            );
                          }}
                          placeholder="Select KPI columns..."
                          className="react-select-container"
                          classNamePrefix="react-select"
                          styles={{
                            control: (base) => ({
                              ...base,
                              borderRadius: "0.5rem",
                              borderColor: "rgb(209 213 219)",
                              boxShadow: "none",
                              "&:hover": {
                                borderColor: "rgb(79 70 229)",
                              },
                              padding: "2px",
                              backgroundColor: "transparent",
                            }),
                            multiValue: (base) => ({
                              ...base,
                              backgroundColor: "rgb(236 253 245)",
                              borderRadius: "0.375rem",
                            }),
                            multiValueLabel: (base) => ({
                              ...base,
                              color: "rgb(5 150 105)",
                              fontWeight: 500,
                            }),
                            multiValueRemove: (base) => ({
                              ...base,
                              color: "rgb(5 150 105)",
                              ":hover": {
                                backgroundColor: "rgb(209 250 229)",
                                color: "rgb(4 120 87)",
                              },
                            }),
                            menu: (base) => ({
                              ...base,
                              borderRadius: "0.5rem",
                              overflow: "hidden",
                              boxShadow:
                                "0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)",
                            }),
                            option: (base, state) => ({
                              ...base,
                              backgroundColor: state.isSelected
                                ? "rgb(16 185 129)"
                                : state.isFocused
                                ? "rgb(236 253 245)"
                                : base.backgroundColor,
                              color: state.isSelected ? "white" : "inherit",
                              ":active": {
                                backgroundColor: state.isSelected
                                  ? "rgb(5 150 105)"
                                  : "rgb(209 250 229)",
                              },
                            }),
                          }}
                          theme={(theme) => ({
                            ...theme,
                            colors: {
                              ...theme.colors,
                              primary: "rgb(16 185 129)",
                              primary75: "rgb(5 150 105)",
                              primary50: "rgb(4 120 87)",
                              primary25: "rgb(209 250 229)",
                            },
                          })}
                        />
                      </div>
                    </div>

                    {/* Important Columns Section */}
                    <div className="mb-8 space-y-3">
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                        Select Important Columns
                      </label>
                      <div className="relative">
                        <Select
                          isMulti
                          menuCloseOnSelect={false}
                          closeMenuOnSelect={false}
                          options={columns
                            .filter(
                              (col) =>
                                !droppedColumns.includes(col.id) &&
                                !selectedKpi.includes(col.id)
                            )
                            .map((col) => ({ value: col.id, label: col.name }))}
                          value={selectedImportant.map((id) => ({
                            value: id,
                            label: columns.find((col) => col.id === id)?.name,
                          }))}
                          onChange={(selected) => {
                            setSelectedImportant(
                              selected.map((option) => option.value)
                            );
                          }}
                          placeholder="Select important columns..."
                          className="react-select-container"
                          classNamePrefix="react-select"
                          styles={{
                            control: (base) => ({
                              ...base,
                              borderRadius: "0.5rem",
                              borderColor: "rgb(209 213 219)",
                              boxShadow: "none",
                              "&:hover": {
                                borderColor: "rgb(79 70 229)",
                              },
                              padding: "2px",
                              backgroundColor: "transparent",
                            }),
                            multiValue: (base) => ({
                              ...base,
                              backgroundColor: "rgb(239 246 255)",
                              borderRadius: "0.375rem",
                            }),
                            multiValueLabel: (base) => ({
                              ...base,
                              color: "rgb(37 99 235)",
                              fontWeight: 500,
                            }),
                            multiValueRemove: (base) => ({
                              ...base,
                              color: "rgb(37 99 235)",
                              ":hover": {
                                backgroundColor: "rgb(219 234 254)",
                                color: "rgb(29 78 216)",
                              },
                            }),
                            menu: (base) => ({
                              ...base,
                              borderRadius: "0.5rem",
                              overflow: "hidden",
                              boxShadow:
                                "0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)",
                            }),
                            option: (base, state) => ({
                              ...base,
                              backgroundColor: state.isSelected
                                ? "rgb(59 130 246)"
                                : state.isFocused
                                ? "rgb(239 246 255)"
                                : base.backgroundColor,
                              color: state.isSelected ? "white" : "inherit",
                              ":active": {
                                backgroundColor: state.isSelected
                                  ? "rgb(37 99 235)"
                                  : "rgb(219 234 254)",
                              },
                            }),
                          }}
                          theme={(theme) => ({
                            ...theme,
                            colors: {
                              ...theme.colors,
                              primary: "rgb(59 130 246)",
                              primary75: "rgb(37 99 235)",
                              primary50: "rgb(29 78 216)",
                              primary25: "rgb(219 234 254)",
                            },
                          })}
                        />
                      </div>
                    </div>

                    {/* Submit Button */}
                    <motion.button
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                      onClick={handleSubmitColumns}
                      disabled={
                        selectedKpi.length === 0 ||
                        selectedImportant.length === 0
                      }
                      className={`w-full px-4 py-3 rounded-md text-white flex items-center justify-center gap-2 transition-all ${
                        selectedKpi.length === 0 ||
                        selectedImportant.length === 0
                          ? "bg-gray-400 cursor-not-allowed opacity-70"
                          : "bg-indigo-600 hover:bg-indigo-700 shadow-sm hover:shadow"
                      }`}
                    >
                      <span>Continue to Analysis</span>
                      <ChevronRight className="h-4 w-4" />
                    </motion.button>

                    {/* Error Message */}
                    {columnError && (
                      <motion.div
                        className="mt-6 p-4 rounded-lg bg-red-50 dark:bg-red-900/20 border-l-4 border-red-500 flex items-start"
                        initial={{ opacity: 0, y: -10 }}
                        animate={{ opacity: 1, y: 0 }}
                      >
                        <AlertCircle className="h-5 w-5 text-red-500 mr-3 mt-0.5 flex-shrink-0" />
                        <div>
                          <h3 className="text-sm font-medium text-red-800 dark:text-red-400">
                            Error
                          </h3>
                          <p className="text-sm text-red-700 dark:text-red-300 mt-1">
                            {columnError}
                          </p>
                        </div>
                      </motion.div>
                    )}

                    {/* Success Message */}
                    {columnSuccess && (
                      <motion.div
                        className="mt-6 p-4 rounded-lg bg-green-50 dark:bg-green-900/20 border-l-4 border-green-500 flex items-start"
                        initial={{ opacity: 0, y: -10 }}
                        animate={{ opacity: 1, y: 0 }}
                      >
                        <CheckCircle className="h-5 w-5 text-green-500 mr-3 mt-0.5 flex-shrink-0" />
                        <div>
                          <h3 className="text-sm font-medium text-green-800 dark:text-green-400">
                            Success
                          </h3>
                          <p className="text-sm text-green-700 dark:text-green-300 mt-1">
                            {columnSuccess}
                          </p>
                        </div>
                      </motion.div>
                    )}
                  </div>
                </motion.div>
              )}
            </div>
          </motion.div>
        )}

        {/* ChatBot Modal */}
        <ChatBot
          key={`${project_id}-${file?.name || "no-file"}-${validated}`}
          isOpen={isChatOpen}
          onClose={() => setIsChatOpen(false)}
        />
      </div>
    </div>
  );
}

export default function Configuration() {
  return (
    <ErrorBoundary>
      <ConfigurationContent />
    </ErrorBoundary>
  );
}
