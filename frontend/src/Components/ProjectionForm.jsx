"use client"

import { useState } from "react"
import { useLocation, useNavigate, useParams } from "react-router-dom"

const ProjectionForm = () => {
  const { project_id } = useParams()
  const importantColumnNames = localStorage.getItem("importantColumnNames")
  const location = useLocation()
  const navigate = useNavigate()
  const { currentLevel, currentPath, activeKPI } = location.state
  const baseUrl = import.meta.env.VITE_BASE_URL
  const [formData, setFormData] = useState({
    no_of_months: 12,
    date_column: "",
    kpi: activeKPI,
    increase_factor: 1,
    zero_value_replacement: 0,
    user_added_vars_list: importantColumnNames,
  })

  const handleInputChange = (e) => {
    const { name, value } = e.target
    setFormData((prev) => ({
      ...prev,
      [name]: name === "user_added_vars_list" ? value.split(",") : value,
    }))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()

    try {
      const response = await fetch(`${baseUrl}/projects/${project_id}/time-series/encode`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          project_id,
          level: currentLevel,
          path: currentPath,
          ...formData,
        }),
      })
      console.log(response)
      console.log(response.url)
    } catch (error) {
      console.error("Error:", error)
      // Handle error appropriately
    }
  }

  return (
    <div className="max-w-2xl mx-auto mt-8 bg-white rounded-lg shadow-lg p-6 ">
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-800">Time Series Projection Configuration</h2>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="space-y-2">
          <label htmlFor="no_of_months" className="block text-sm font-medium text-gray-700">
            Number of Months to Predict
          </label>
          <input
            id="no_of_months"
            name="no_of_months"
            type="number"
            value={formData.no_of_months}
            onChange={handleInputChange}
            min="1"
            required
            className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          />
        </div>

        <div className="space-y-2">
          <label htmlFor="date_column" className="block text-sm font-medium text-gray-700">
            Date Column
          </label>
          <input
            id="date_column"
            name="date_column"
            type="text"
            value={formData.date_column}
            onChange={handleInputChange}
            className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          />
        </div>

        <div className="space-y-2">
          <label htmlFor="increase_factor" className="block text-sm font-medium text-gray-700">
            Increase Factor
          </label>
          <input
            id="increase_factor"
            name="increase_factor"
            type="number"
            value={formData.increase_factor}
            onChange={handleInputChange}
            min="1"
            required
            className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          />
        </div>

        <div className="space-y-2">
          <label htmlFor="zero_value_replacement" className="block text-sm font-medium text-gray-700">
            Zero Value Replacement
          </label>
          <input
            id="zero_value_replacement"
            name="zero_value_replacement"
            type="number"
            value={formData.zero_value_replacement}
            onChange={handleInputChange}
            required
            className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          />
        </div>

        <div className="space-y-2">
          <label htmlFor="user_added_vars_list" className="block text-sm font-medium text-gray-700">
            Additional Variables (comma-separated)
          </label>
          <input
            id="user_added_vars_list"
            name="user_added_vars_list"
            type="text"
            value={formData.user_added_vars_list}
            onChange={handleInputChange}
            className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          />
        </div>

        <button
          type="submit"
          className="w-full bg-indigo-500 text-white py-2 px-4 rounded-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 transition-colors"
        >
          Generate Projection
        </button>
      </form>
    </div>
  )
}

export default ProjectionForm

