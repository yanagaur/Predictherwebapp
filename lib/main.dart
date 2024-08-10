import 'package:flutter/material.dart';
import 'package:predicther/screens/home_screen.dart';
import 'package:predicther/screens/splash_screen.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  bool _isDarkMode = false;

  void _toggleTheme(bool isDarkMode) {
    setState(() {
      _isDarkMode = isDarkMode;
    });
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'PredictHer',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        brightness: _isDarkMode ? Brightness.dark : Brightness.light,
      ),

      initialRoute: '/',
      routes: {
        '/': (context) => SplashScreen(),
        '/home': (context) => HomeScreen(toggleTheme: _toggleTheme, isDarkMode: _isDarkMode),
      },
    );
  }
}
