import React from 'react';
import { Link, useLocation } from 'react-router-dom';

const Navbar = ({ appName = "ModelForge" }) => {
  // Get current location
  const location = useLocation();
  
  // Determine if we're on the loading page or settings page where navigation should be limited
  const disableNavigation = location.pathname === '/finetune/load_settings' || 
                            location.pathname === '/finetune/loading';
  
  // Only show the "Get Started" button on the home/landing page
  const showGetStartedButton = location.pathname === '/' || location.pathname === '/home';
  
  // Show "Back to Home" button on pages that aren't home or disabled navigation pages
  const showBackButton = !showGetStartedButton && !disableNavigation;
  
  // Check if we're on the technical details page
  const isTechnicalPage = location.pathname === '/technical-details';
  
  // Define page titles based on route
  const getPageTitle = () => {
    switch (location.pathname) {
      case '/':
      case '/home':
        return null; // No title for home page, it has its own hero section
      case '/finetune/detect':
        return 'Hardware Detection';
      case '/finetune/load_settings':
        return 'Finetuning Settings';
      case '/finetune/loading':
        return 'Training Progress';
      case '/technical-details':
        return 'Technical Documentation';
      case '/app':
        return 'Application';
      case '/models':
        return 'My models'
      default:
        return null;
    }
  };
  
  const pageTitle = getPageTitle();
  const showHeaderContent = pageTitle && !isTechnicalPage; // Show header content except on home and tech details page
  
  // Header section with dynamic title - use dark blue bg except for home
  const renderHeader = () => {
    // Skip rendering the header on pages that already have their own title sections
    // or that show duplicate headers
    if (location.pathname === '/' || 
        location.pathname === '/home' || 
        location.pathname === '/finetune/load_settings' ||
        location.pathname === '/finetune/detect') {
      return null;
    }
    
    // For technical details page, render a custom header that matches the screenshot
    if (location.pathname === '/technical-details') {
      return (
        <div className="bg-[#141b2d] pt-6 pb-12">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="text-center">
              <h1 className="text-4xl md:text-5xl font-bold mb-4 text-white">Technical Documentation</h1>
              <p className="text-xl text-gray-300 max-w-3xl mx-auto">
                Comprehensive technical details for developers and ML engineers
              </p>
            </div>
          </div>
        </div>
      );
    }
    
    return (
      <header className="py-6 md:py-12 bg-[#141b2d]">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          {showHeaderContent && (
            <div className="mt-8 text-center">
              <h1 className="text-3xl md:text-4xl font-bold mb-4 text-white">{pageTitle}</h1>
              {/* Removed all specific page descriptions since they're duplicating */}
            </div>
          )}
        </div>
      </header>
    );
  };
  
  // Removed unused renderFooter helper (was returning null and never used)

  return (
    <>
      {/* Main Navigation Bar - Black for home page, dark blue for other pages */}
      <nav className={`${location.pathname === '/' || location.pathname === '/home' ? 'bg-black' : 'bg-[#141b2d]'} py-4 px-4`}>
        <div className="container mx-auto flex justify-between items-center">
          <div className="flex items-center">
            {disableNavigation ? (
              // When in settings/loading page, logo is just text, not a link
              <div className="flex items-center">
                <div className="w-8 h-8 mr-2">
                  <svg width="62" height="62" viewBox="0 0 62 62" fill="none" xmlns="http://www.w3.org/2000/svg" className="w-full h-full">
                    <rect width="62" height="62" rx="10" fill="#E34D26"/>
                    <path d="M31 19.5V24.5M31 19.5H26M31 19.5H36M31 42.5V37.5M31 42.5H36M31 42.5H26M19.5 31H24.5M19.5 31V26M19.5 31V36M42.5 31H37.5M42.5 31V36M42.5 31V26M26 19.5V24.5H24.5V26H19.5M36 19.5V24.5H37.5V26H42.5M26 42.5V37.5H24.5V36H19.5M36 42.5V37.5H37.5V36H42.5" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                </div>
                <span className="text-white text-xl font-bold">{appName}</span>
              </div>
            ) : (
              // Otherwise, logo is a link to home
              <Link to="/" className="flex items-center">
                <div className="w-8 h-8 mr-2">
                  <svg width="62" height="62" viewBox="0 0 62 62" fill="none" xmlns="http://www.w3.org/2000/svg" className="w-full h-full">
                    <rect width="62" height="62" rx="10" fill="#E34D26"/>
                    <path d="M31 19.5V24.5M31 19.5H26M31 19.5H36M31 42.5V37.5M31 42.5H36M31 42.5H26M19.5 31H24.5M19.5 31V26M19.5 31V36M42.5 31H37.5M42.5 31V36M42.5 31V26M26 19.5V24.5H24.5V26H19.5M36 19.5V24.5H37.5V26H42.5M26 42.5V37.5H24.5V36H19.5M36 42.5V37.5H37.5V36H42.5" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                </div>
                <span className="text-white text-xl font-bold">{appName}</span>
              </Link>
            )}
          </div>

          {/* Right side content based on screenshot */}
          <div className="flex items-center">
            {/* My Models tab */}
            {!disableNavigation && (
              <Link
                to="/models"
                className={`text-white px-4 py-2 rounded-lg transition ${
                  location.pathname === '/models'
                    ? 'bg-[#232c40] font-semibold'
                    : 'hover:bg-[#232c40]'
                }`}
                style={{ marginRight: '0.5rem' }}
              >
                My Models
              </Link>
            )}
            {/* Back to Home button on inner pages */}
            {showBackButton && (
              <Link to="/" className="bg-[#232c40] hover:bg-[#2d3a52] text-white px-4 py-2 rounded-lg flex items-center gap-2 transition">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 19l-7-7 7-7" />
                </svg>
                Back to Home
              </Link>
            )}
            
            {/* Get Started button only on home page */}
            {showGetStartedButton && (
              <Link to="/finetune/detect" className="text-orange-500 hover:text-orange-400 px-4 py-2 transition">
                Get Started
              </Link>
            )}
          </div>
        </div>
      </nav>
      
      {/* Render the header with dynamic title */}
      {renderHeader()}
      
      {/* The children components will be rendered in between by App.js */}
      
      {/* Styling for hiding scrollbars but keeping functionality */}
      <style jsx="true">{`
        .bg-technical-tabs {
          background-color: #182035;
        }
      `}</style>
    </>
  );
};

export default Navbar;