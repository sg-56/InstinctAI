"use client";

import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import { Home, AlertTriangle } from "lucide-react";
import { getAuthState } from "../utils/auth";

const NoPage = () => {
  const { com_id } = getAuthState();

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 flex flex-col justify-center py-12 sm:px-6 lg:px-8">
      <motion.div
        className="sm:mx-auto sm:w-full sm:max-w-md text-center"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <div className="mx-auto flex items-center justify-center h-24 w-24 rounded-full bg-yellow-100 dark:bg-yellow-900/30">
          <AlertTriangle className="h-12 w-12 text-yellow-600 dark:text-yellow-500" />
        </div>

        <motion.h1
          className="mt-6 text-6xl font-extrabold text-gray-900 dark:text-white"
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ delay: 0.2, duration: 0.5 }}
        >
          404
        </motion.h1>

        <motion.h2
          className="mt-2 text-3xl font-bold text-gray-900 dark:text-white"
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.3, duration: 0.5 }}
        >
          Page not found
        </motion.h2>

        <motion.p
          className="mt-4 text-base text-gray-600 dark:text-gray-400 max-w-md mx-auto"
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.4, duration: 0.5 }}
        >
          Sorry, we couldn&apos;t find the page you&apos;re looking for. The
          page might have been moved, deleted, or never existed.
        </motion.p>

        <motion.div
          className="mt-8"
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.5, duration: 0.5 }}
        >
          <Link
            to={com_id ? `/${com_id}` : "/login"}
            className="inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
          >
            <Home className="-ml-1 mr-2 h-5 w-5" />
            {com_id ? "Back to Home" : "Go to Login"}
          </Link>
        </motion.div>
      </motion.div>

      <motion.div
        className="mt-12 sm:mx-auto sm:w-full sm:max-w-md"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.6, duration: 0.5 }}
      >
        <div className="bg-white dark:bg-gray-800 py-8 px-4 shadow sm:rounded-lg sm:px-10">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
            Looking for something?
          </h3>
          <ul className="space-y-3 text-sm">
            <li>
              <Link
                to={com_id ? `/${com_id}` : "/login"}
                className="text-indigo-600 hover:text-indigo-500 dark:text-indigo-400 dark:hover:text-indigo-300 flex items-center"
              >
                <span className="mr-2">→</span>
                {com_id ? "Go to the home page" : "Go to login page"}
              </Link>
            </li>
            {!com_id && (
              <>
                <li>
                  <Link
                    to="/login"
                    className="text-indigo-600 hover:text-indigo-500 dark:text-indigo-400 dark:hover:text-indigo-300 flex items-center"
                  >
                    <span className="mr-2">→</span> Sign in to your account
                  </Link>
                </li>
                <li>
                  <Link
                    to="/signup"
                    className="text-indigo-600 hover:text-indigo-500 dark:text-indigo-400 dark:hover:text-indigo-300 flex items-center"
                  >
                    <span className="mr-2">→</span> Create a new account
                  </Link>
                </li>
              </>
            )}
          </ul>
        </div>
      </motion.div>
    </div>
  );
};

export default NoPage;
