import React from 'react';
import { Link } from 'react-router-dom';
import '../styles/LandingPage.css';

const LandingPage = ({ appName = "ModelForge" }) => {
  const features = [
    {
      title: "No-Code Interface",
      description: "Train AI models with simple clicks - no programming required",
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="w-10 h-10" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
        </svg>
      )
    },
    {
      title: "Smart Hardware Detection",
      description: "Automatically configures for your computer's capabilities",
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="w-10 h-10" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2z" />
        </svg>
      )
    },
    {
      title: "Efficient Training",
      description: "Get results in minutes, not days",
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="w-10 h-10" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
        </svg>
      )
    },
    {
      title: "Business Ready",
      description: "Custom AI models tailored to your industry needs",
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="w-10 h-10" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 13.255A23.931 23.931 0 0112 15c-3.183 0-6.22-.62-9-1.745M16 6V4a2 2 0 00-2-2h-4a2 2 0 00-2 2v2m4 6h.01M5 20h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
        </svg>
      )
    }
  ];

  return (
    <div className="bg-black text-gray-100 min-h-screen">
      {/* Hero Section with Animation */}
      <div className="hero-section py-16 lg:py-24 relative overflow-hidden">
        {/* Animated background elements */}
        <div className="absolute inset-0 z-0">
          <div className="neural-network">
            <div className="node node-1"></div>
            <div className="node node-2"></div>
            <div className="node node-3"></div>
            <div className="node node-4"></div>
            <div className="node node-5"></div>
            <div className="node node-6"></div>
            <div className="connection connection-1"></div>
            <div className="connection connection-2"></div>
            <div className="connection connection-3"></div>
            <div className="connection connection-4"></div>
            <div className="connection connection-5"></div>
          </div>
        </div>
        
        <div className="max-w-6xl mx-auto px-4 relative z-10">
          <div className="animate-fade-in text-center">
            <div className="mb-6 inline-block relative">
              <span className="bg-orange-500 text-white text-sm md:text-base px-3 py-1 rounded-full transform -rotate-2 inline-block">
                No Coding Required!
              </span>
            </div>
            
            <h1 className="text-4xl md:text-6xl lg:text-7xl font-bold mb-6 relative">
              <span className="block">AI So Easy</span>
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-orange-400 to-orange-600">
                Your Grandma Can Use It
              </span>
              <div className="absolute -top-8 -right-8 md:right-64 transform rotate-12 opacity-70">
                <svg className="w-16 h-16 text-orange-500" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M5 2a1 1 0 011 1v1h1a1 1 0 010 2H6v1a1 1 0 01-2 0V6H3a1 1 0 010-2h1V3a1 1 0 011-1zm0 10a1 1 0 011 1v1h1a1 1 0 110 2H6v1a1 1 0 11-2 0v-1H3a1 1 0 110-2h1v-1a1 1 0 011-1zM12 2a1 1 0 01.967.744L14.146 7.2 17.5 9.134a1 1 0 010 1.732l-3.354 1.935-1.18 4.455a1 1 0 01-1.933 0L9.854 12.8 6.5 10.866a1 1 0 010-1.732l3.354-1.935 1.18-4.455A1 1 0 0112 2z" />
                </svg>
              </div>
            </h1>
            
            <p className="text-xl md:text-2xl text-gray-300 mb-8 max-w-3xl mx-auto">
              Transform and customize AI models in minutes with zero technical knowledge.
              Point, click, and watch your custom AI come to life!
            </p>

            <div className="flex flex-wrap justify-center gap-4">
              <Link to="/finetune/select-provider" className="group bg-gradient-to-r from-orange-500 to-orange-600 hover:from-orange-600 hover:to-orange-700 px-8 py-4 rounded-lg text-lg font-medium transition shadow-lg hover:shadow-orange-500/30 transform hover:-translate-y-1">
                Start Building Your AI
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 inline-block ml-2 transition-transform group-hover:translate-x-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 7l5 5m0 0l-5 5m5-5H6" />
                </svg>
              </Link>
              <Link to="/technical-details" className="border border-orange-500 text-orange-400 hover:bg-orange-500 hover:text-white px-8 py-4 rounded-lg text-lg font-medium transition transform hover:-translate-y-1">
                Learn More
              </Link>
            </div>
          </div>

          {/* Feature Cards in Hero Section */}
          <div className="mt-16 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {features.map((feature, index) => (
              <div 
                key={feature.title} 
                className="feature-card bg-gray-900/70 backdrop-blur-sm border border-gray-800 p-6 rounded-xl transform transition-all duration-300 hover:scale-105 hover:border-orange-500 hover:shadow-lg hover:shadow-orange-500/20"
                style={{ animationDelay: `${index * 0.1}s` }}
              >
                <div className="icon-container mb-4 p-3 bg-orange-500/10 rounded-lg inline-block">
                  <div className="text-orange-500">
                    {feature.icon}
                  </div>
                </div>
                <h3 className="text-xl font-bold mb-2">{feature.title}</h3>
                <p className="text-gray-400">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* How It Works Section */}
      <div id="how-it-works" className="how-it-works py-16 lg:py-24 bg-gradient-to-b from-black to-gray-900">
        <div className="max-w-6xl mx-auto px-4">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">From Zero to AI Hero in 3 Simple Steps</h2>
            <p className="text-xl text-gray-300 max-w-2xl mx-auto">
              No PhD required. Just point, click, and watch the magic happen.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {/* Step 1 */}
            <div className="step-card relative">
              <div className="absolute -left-4 -top-4 w-14 h-14 bg-orange-500 rounded-full flex items-center justify-center text-2xl font-bold">
                1
              </div>
              <div className="bg-gray-800 rounded-xl p-8 pt-12 h-full transform transition-all duration-300 hover:scale-105 hover:shadow-xl hover:shadow-orange-500/10">
                <h3 className="text-2xl font-bold mb-4">Hardware Detection</h3>
                <p className="text-gray-300 mb-4">
                  Our system automatically scans your computer and configures the perfect settings based on your hardware.
                </p>
                <div className="mt-4 flex justify-center">
                  <div className="w-16 h-16 rounded-full bg-orange-500/20 flex items-center justify-center">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-orange-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2z" />
                    </svg>
                  </div>
                </div>
              </div>
            </div>

            {/* Step 2 */}
            <div className="step-card relative">
              <div className="absolute -left-4 -top-4 w-14 h-14 bg-orange-500 rounded-full flex items-center justify-center text-2xl font-bold">
                2
              </div>
              <div className="bg-gray-800 rounded-xl p-8 pt-12 h-full transform transition-all duration-300 hover:scale-105 hover:shadow-xl hover:shadow-orange-500/10">
                <h3 className="text-2xl font-bold mb-4">Pick Your Settings</h3>
                <p className="text-gray-300 mb-4">
                  Use our simple sliders and dropdowns to customize your AI. No code or complex terms to worry about.
                </p>
                <div className="mt-4 flex justify-center">
                  <div className="w-16 h-16 rounded-full bg-orange-500/20 flex items-center justify-center">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-orange-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                    </svg>
                  </div>
                </div>
              </div>
            </div>

            {/* Step 3 */}
            <div className="step-card relative">
              <div className="absolute -left-4 -top-4 w-14 h-14 bg-orange-500 rounded-full flex items-center justify-center text-2xl font-bold">
                3
              </div>
              <div className="bg-gray-800 rounded-xl p-8 pt-12 h-full transform transition-all duration-300 hover:scale-105 hover:shadow-xl hover:shadow-orange-500/10">
                <h3 className="text-2xl font-bold mb-4">Launch Your AI</h3>
                <p className="text-gray-300 mb-4">
                  Click one button and watch as your custom AI model trains and deploys automatically.
                </p>
                <div className="mt-4 flex justify-center">
                  <div className="w-16 h-16 rounded-full bg-orange-500/20 flex items-center justify-center">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-orange-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Business Section */}
      <div className="business-section py-16 lg:py-24 bg-gradient-to-b from-gray-900 to-black">
        <div className="max-w-6xl mx-auto px-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
            <div className="order-2 lg:order-1">
              <h2 className="text-3xl md:text-4xl font-bold mb-6">Perfect for Business</h2>
              <p className="text-xl text-gray-300 mb-8">
                Skip the expensive AI consultants. Train models on your business data in-house with zero technical expertise needed.
              </p>
              
              <ul className="space-y-4">
                {[
                  "Automate customer support with custom chatbots",
                  "Extract insights from your business documents",
                  "Create AI assistants trained on your company knowledge",
                  "Generate content aligned with your brand voice"
                ].map((item, index) => (
                  <li key={index} className="flex items-start">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 mr-2 text-orange-500 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                    <span className="text-gray-300">{item}</span>
                  </li>
                ))}
              </ul>
              
              <div className="mt-8">
                <Link to="/finetune/select-provider" className="group bg-gradient-to-r from-orange-500 to-orange-600 hover:from-orange-600 hover:to-orange-700 px-8 py-4 rounded-lg text-lg font-medium transition shadow-lg hover:shadow-orange-500/30">
                  Start Your Business AI
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 inline-block ml-2 transition-transform group-hover:translate-x-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 7l5 5m0 0l-5 5m5-5H6" />
                  </svg>
                </Link>
              </div>
            </div>
            
            <div className="order-1 lg:order-2 flex justify-center">
              <div className="business-illustration relative w-full max-w-md">
                <div className="relative bg-gray-800 rounded-xl overflow-hidden shadow-2xl transform rotate-2 hover:rotate-0 transition-all duration-500">
                  <div className="aspect-w-16 aspect-h-9 p-6">
                    <div className="bg-gray-700 w-full h-32 rounded-lg mb-4 animate-pulse"></div>
                    <div className="bg-gray-700 w-3/4 h-8 rounded-lg mb-3 animate-pulse"></div>
                    <div className="bg-gray-700 w-1/2 h-8 rounded-lg mb-6 animate-pulse"></div>
                    <div className="flex space-x-4">
                      <div className="bg-orange-500 w-1/3 h-10 rounded-lg animate-pulse"></div>
                      <div className="bg-gray-700 w-1/3 h-10 rounded-lg animate-pulse"></div>
                    </div>
                  </div>
                </div>
                <div className="absolute -bottom-6 -right-6 bg-orange-500 rounded-full w-16 h-16 flex items-center justify-center text-white text-xl font-bold transform rotate-12">
                  New!
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* CTA Section */}
      <div className="cta-section py-16 lg:py-24 bg-black text-center relative overflow-hidden">
        <div className="absolute inset-0 z-0 opacity-20">
          {/* Background decoration */}
          <div className="absolute top-0 left-1/4 w-64 h-64 bg-orange-500 rounded-full filter blur-3xl"></div>
          <div className="absolute bottom-0 right-1/4 w-64 h-64 bg-orange-600 rounded-full filter blur-3xl"></div>
        </div>
        
        <div className="max-w-4xl mx-auto px-4 relative z-10">
          <h2 className="text-3xl md:text-5xl font-bold mb-6">Ready to Build Your Custom AI?</h2>
          <p className="text-xl text-gray-300 mb-8 max-w-2xl mx-auto">
            If your grandma can use a smartphone, she can build an AI with ModelForge.
            No coding. No technical skills. Just pure AI power at your fingertips.
          </p>
          
          <Link to="/finetune/select-provider" className="group bg-gradient-to-r from-orange-500 to-orange-600 hover:from-orange-600 hover:to-orange-700 px-10 py-5 rounded-lg text-xl font-bold transition shadow-xl hover:shadow-orange-500/30 transform hover:-translate-y-1">
            Start Building Now
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 inline-block ml-2 transition-transform group-hover:translate-x-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 7l5 5m0 0l-5 5m5-5H6" />
            </svg>
          </Link>
          
          <p className="mt-6 text-gray-400">
            No credit card required. Ready in minutes.
          </p>
          
          <div className="mt-12">
            <Link to="/technical-details" className="text-orange-500 hover:text-orange-400 flex items-center justify-center group">
              <span>View technical specifications</span>
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 ml-2 group-hover:translate-x-1 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
              </svg>
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LandingPage;