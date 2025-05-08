"use client"

import { useEffect, useState } from "react"
import { useLocation, useNavigate, useParams } from "react-router-dom"
import Select from "react-select"
import axios from "axios"
import { AlertCircle, ArrowRight } from "lucide-react"
import { motion } from "framer-motion"
import { useDispatch } from "react-redux"
import { updateProject } from "../redux/projectsSlice"

const CustomOption = ({ data, innerRef, innerProps, isSelected, selectOption, deselectOption }) => {
  const handleCheckboxChange = (e) => {
    e.stopPropagation()
    if (isSelected) {
      deselectOption(data)
    } else {
      selectOption(data)
    }
  }

  return (
    <motion.div
      ref={innerRef}
      {...innerProps}
      className="flex items-center gap-2 px-3 py-2 cursor-pointer hover:bg-gray-100"
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
    >
      <input
        type="checkbox"
        checked={isSelected}
        onChange={handleCheckboxChange}
        className="h-4 w-4 rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
        onClick={(e) => e.stopPropagation()}
      />
      <span className="text-sm font-medium text-gray-700">{data.label}</span>
    </motion.div>
  )
}

const SelectColumns = () => {
  const location = useLocation()
  const { project_id } = useParams()
  const [columns, setColumns] = useState([])
  const [loading, setLoading] = useState(true)
  const [droppedColumns, setDroppedColumns] = useState([])
  const [selectedKpi, setSelectedKpi] = useState([])
  const [selectedImportant, setSelectedImportant] = useState([])
  const [errors, setErrors] = useState({})
  const [success, setSuccess] = useState("")
  const baseUrl = import.meta.env.VITE_BASE_URL
  const navigate = useNavigate()
  const dispatch = useDispatch()

  const fetchColumns = async () => {
    if (!project_id) {
      console.error("No project_id found")
      setLoading(false)
      return
    }

    try {
      const response = await axios.post(`${baseUrl}/projects/${project_id}/dataset/columns`, { project_id })
      setColumns(response.data.columns)
      localStorage.setItem("columns", JSON.stringify(response.data.columns))
      dispatch(updateProject({ projectId: project_id, columns: response.data.columns }))
      setLoading(false)
    } catch (error) {
      console.error("Error fetching columns:", error)
      setErrors({ fetch: "Failed to fetch columns. Please try again." })
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchColumns()
  }, [baseUrl, project_id])

  const options = columns.map((col) => ({
    value: col,
    label: col,
  }))

  const handleDrop = async () => {
    if (droppedColumns.length > 0) {
      try {
        setLoading(true)
        const droppedColumnNames = droppedColumns.map((col) => col.value)
        await axios.post(`${baseUrl}/projects/${project_id}/dataset/columns/drop`, {
          project_id,
          column: droppedColumnNames,
        })
        dispatch(updateProject({ projectId: project_id, droppedColumns: droppedColumnNames }))
        await fetchColumns()
        setSuccess("Columns dropped successfully!")
        localStorage.setItem("droppedColumns", JSON.stringify(droppedColumnNames))
      } catch (error) {
        console.error("Error submitting data:", error.response?.data?.message)
        setErrors({ drop: "Failed to drop columns. Please try again." })
      } finally {
        setLoading(false)
      }
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (selectedKpi.length === 0 || selectedImportant.length === 0) {
      setErrors({ submit: "Please select both KPI and Important columns before submitting." })
      return
    }
    const importantColumnNames = selectedImportant.map((col) => col.value)
    const kpiList = selectedKpi.map((kpi) => kpi.value)
    localStorage.setItem("importantColumnNames", JSON.stringify(importantColumnNames))
    localStorage.setItem("kpiList", JSON.stringify(kpiList))

    // Dispatch updateProject action to update the importantColumnNames and kpiList in the project
    dispatch(
      updateProject({
        projectId: project_id,
        importantColumnNames,
        kpiList,
      }),
    )

    navigate(`/projects/${project_id}/select-kpi`, {
      state: { project_id, importantColumnNames, kpiList },
    })
  }

  const handleDroppedColumnsChange = (selected) => {
    const uniqueSelected = selected.filter(
      (item, index, self) => index === self.findIndex((t) => t.value === item.value),
    )
    setDroppedColumns(uniqueSelected)
  }

  if (loading) {
    return (
      <div className="flex justify-center items-center h-[30vh]">
        <div className="animate-spin rounded-full h-32 w-32 border-t-2 border-b-2 border-indigo-500"></div>
      </div>
    )
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-3xl mx-auto bg-white shadow-lg rounded-lg">
        <div className="p-6 space-y-6">
          <h2 className="text-2xl font-bold text-gray-800 mb-4">Select Columns</h2>

          {errors.fetch && (
            <div className="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 mb-4" role="alert">
              <p className="font-bold">Error</p>
              <p>{errors.fetch}</p>
            </div>
          )}

          {success && (
            <div className="bg-green-100 border-l-4 border-green-500 text-green-700 p-4 mb-4" role="alert">
              <p className="font-bold">Success</p>
              <p>{success}</p>
            </div>
          )}

          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Select Columns to Drop</label>
              <div className="flex items-center gap-4">
                <div className="flex-grow">
                  <Select
                    value={droppedColumns}
                    onChange={handleDroppedColumnsChange}
                    options={options}
                    closeMenuOnSelect={false}
                    isMulti={true}
                    placeholder="Select columns to drop"
                    className="react-select-container"
                    classNamePrefix="react-select"
                  />
                </div>
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className={`px-4 py-2 bg-red-500 text-white rounded-md flex items-center justify-center gap-2 transition duration-300 ease-in-out ${
                    droppedColumns.length === 0 ? "opacity-50 cursor-not-allowed" : "hover:bg-red-600"
                  }`}
                  onClick={handleDrop}
                  disabled={droppedColumns.length === 0}
                >
                  Drop
                </motion.button>
              </div>
            </div>

            {success && (
              <>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Select KPI Columns</label>
                  <Select
                    value={selectedKpi}
                    onChange={(selected) => {
                      const uniqueSelected = selected.filter(
                        (item, index, self) => index === self.findIndex((t) => t.value === item.value),
                      )
                      setSelectedKpi(uniqueSelected)
                    }}
                    options={options}
                    closeMenuOnSelect={false}
                    isMulti={true}
                    placeholder="Select KPI columns"
                    className="react-select-container"
                    classNamePrefix="react-select"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Select Important Columns</label>
                  <Select
                    value={selectedImportant}
                    onChange={(selected) => {
                      const uniqueSelected = selected.filter(
                        (item, index, self) => index === self.findIndex((t) => t.value === item.value),
                      )
                      setSelectedImportant(uniqueSelected)
                    }}
                    options={options}
                    isMulti={true}
                    closeMenuOnSelect={false}
                    placeholder="Select important columns"
                    className="react-select-container"
                    classNamePrefix="react-select"
                  />
                </div>
              </>
            )}
          </div>

          {errors.submit && (
            <div className="flex items-center gap-2 text-red-600 mt-4">
              <AlertCircle size={18} />
              <span>{errors.submit}</span>
            </div>
          )}

          <div className="mt-8 flex justify-end">
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={handleSubmit}
              disabled={!success}
              className="px-6 py-2 bg-indigo-600 text-white font-semibold rounded-md hover:bg-indigo-700 transition duration-300 ease-in-out flex items-center gap-2"
              type="button"
            >
              Submit <ArrowRight size={18} />
            </motion.button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default SelectColumns

