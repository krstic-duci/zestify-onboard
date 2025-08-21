'use client';

import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { Textarea } from '@/components/ui/textarea';
import React, { useEffect, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import rehypeHighlight from 'rehype-highlight';
import remarkGfm from 'remark-gfm';

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  error?: string;
}

export default function Chat() {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [showScrollButton, setShowScrollButton] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new messages arrive
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // Handle scroll detection for showing scroll button
  const handleScroll = () => {
    if (scrollContainerRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = scrollContainerRef.current;
      const isNearBottom = scrollHeight - scrollTop - clientHeight < 150;
      setShowScrollButton(!isNearBottom);
    }
  };

  // Auto-scroll when messages change or loading state changes
  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const sendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!input.trim() || isLoading) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: input.trim(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          messages: [{ role: 'user', content: input.trim() }],
        }),
      });

      const data = await response.json();

      const assistantMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.content || data.response || 'No response received.',
        error: data.error,
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      const errorMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: 'Failed to get response from the server.',
        error: error instanceof Error ? error.message : 'Unknown error',
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-white p-4">
      <div className="max-w-4xl mx-auto space-y-6">
        
        {/* Header */}
        <Card className="border-gray-100">
          <CardHeader className="text-center">
            <h1 className="text-4xl font-bold text-gray-900">
              Zestify Knowledge Base
            </h1>
            <p className="text-gray-600 text-lg">
              Ask questions about the Zestify codebase and get AI-powered answers.
            </p>
          </CardHeader>
        </Card>

        {/* Chat Messages */}
        <Card className="border-gray-100 relative">
          <CardContent className="p-6">
            <div 
              ref={scrollContainerRef}
              className="h-96 overflow-y-auto space-y-4"
              onScroll={handleScroll}
            >
              {messages.length === 0 && (
                <div className="text-center py-12">
                  <h3 className="text-xl font-semibold text-gray-900 mb-4">Welcome to Zestify AI Assistant!</h3>
                  <p className="text-gray-600 mb-6">Start by asking a question about the Zestify codebase.</p>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-3 max-w-2xl mx-auto">
                    {[
                      "How does authentication work?",
                      "What's the database structure?", 
                      "How do I add a new recipe?"
                    ].map((question, idx) => (
                      <Button 
                        key={idx}
                        variant="outline"
                        onClick={() => setInput(question)}
                        className="h-auto p-3 text-sm hover:bg-gray-50 border-gray-200"
                      >
                        "{question}"
                      </Button>
                    ))}
                  </div>
                </div>
              )}

              {messages.map((message) => (
                <div key={message.id} className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'} gap-3`}>
                  
                  {/* Assistant Avatar */}
                  {message.role === 'assistant' && (
                    <Avatar>
                      <AvatarFallback className="bg-gray-100 text-gray-700">
                        🤖
                      </AvatarFallback>
                    </Avatar>
                  )}

                  {/* Message */}
                  <div className={`max-w-[80%] ${message.role === 'user' ? 'order-1' : ''}`}>
                    <Card className={message.role === 'user' ? 'bg-gray-900 text-white border-gray-900' : 'border-gray-200'}>
                      <CardContent className="p-4">
                        {message.role === 'assistant' ? (
                          <div className="prose prose-sm max-w-none">
                            <ReactMarkdown 
                              remarkPlugins={[remarkGfm]}
                              rehypePlugins={[rehypeHighlight]}
                              components={{
                                // Custom styling for code blocks
                                code({children, ...props}) {
                                  const isInline = !props.className?.includes('language-');
                                  return isInline ? (
                                    <code className="bg-gray-100 px-1 py-0.5 rounded text-sm font-mono" {...props}>
                                      {children}
                                    </code>
                                  ) : (
                                    <code className="block bg-gray-100 p-3 rounded-lg text-sm font-mono overflow-x-auto" {...props}>
                                      {children}
                                    </code>
                                  );
                                },
                                // Custom styling for lists
                                ul: ({children}) => (
                                  <ul className="list-disc pl-6 space-y-1">{children}</ul>
                                ),
                                ol: ({children}) => (
                                  <ol className="list-decimal pl-6 space-y-1">{children}</ol>
                                ),
                                h1: ({children}) => (
                                  <h1 className="text-lg font-bold mb-3">{children}</h1>
                                ),
                                h2: ({children}) => (
                                  <h2 className="text-base font-bold mb-2">{children}</h2>
                                ),
                                h3: ({children}) => (
                                  <h3 className="text-sm font-bold mb-2">{children}</h3>
                                ),
                                p: ({children}) => (
                                  <p className="mb-3 last:mb-0 leading-relaxed">{children}</p>
                                ),
                              }}
                            >
                              {message.content}
                            </ReactMarkdown>
                          </div>
                        ) : (
                          <div className="leading-relaxed">
                            {message.content}
                          </div>
                        )}
                        
                        {message.error && (
                          <div className="mt-3 p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
                            <strong>Error:</strong> {message.error}
                          </div>
                        )}
                      </CardContent>
                    </Card>
                  </div>

                  {/* User Avatar */}
                  {message.role === 'user' && (
                    <Avatar>
                      <AvatarFallback className="bg-gray-700 text-white">
                        👤
                      </AvatarFallback>
                    </Avatar>
                  )}
                </div>
              ))}
              
              {/* Loading */}
              {isLoading && (
                <div className="flex justify-start gap-3">
                  <Avatar>
                    <AvatarFallback className="bg-gray-100 text-gray-700">
                      🤖
                    </AvatarFallback>
                  </Avatar>
                  <Card className="border-gray-200">
                    <CardContent className="p-4">
                      <div className="flex items-center gap-2">
                        <div className="flex space-x-1">
                          <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce"></div>
                          <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                          <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                        </div>
                        <span className="text-sm text-gray-600">Thinking...</span>
                      </div>
                    </CardContent>
                  </Card>
                </div>
              )}
              
              {/* Invisible element to scroll to */}
              <div ref={messagesEndRef} />
            </div>
            
            {/* Scroll to bottom button */}
            {showScrollButton && (
              <Button
                onClick={scrollToBottom}
                className="absolute bottom-20 right-6 bg-gray-900 hover:bg-gray-800 text-white rounded-full w-12 h-12 p-0 shadow-lg"
                aria-label="Scroll to bottom"
              >
                ↓
              </Button>
            )}
          </CardContent>
        </Card>

        {/* Input Form */}
        <Card className="border-gray-200">
          <CardContent className="p-6">
            <form onSubmit={sendMessage} className="space-y-4">
              <Textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask me anything about the Zestify codebase...

For example:
• How does the authentication system work?
• What's the database schema and structure?
• How do I add a new recipe feature?"
                disabled={isLoading}
                className="min-h-[120px] resize-y border-gray-300 focus:border-gray-500"
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
                    e.preventDefault();
                    sendMessage(e);
                  }
                }}
              />
              <div className="flex justify-between items-center">
                <div className="text-sm text-gray-600 flex items-center gap-2">
                  <Badge variant="outline" className="border-gray-300">⌘ + Enter</Badge>
                  <span> to send</span>
                </div>
                <Button 
                  type="submit" 
                  disabled={isLoading || !input.trim()}
                  className="bg-gray-900 hover:bg-gray-800 text-white"
                >
                  {isLoading ? (
                    <div className="flex items-center gap-2">
                      <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                      <span>Sending...</span>
                    </div>
                  ) : (
                    <div className="flex items-center gap-2">
                      <span>Send Question</span>
                    </div>
                  )}
                </Button>
              </div>
            </form>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
