import 'package:flutter/material.dart';
import 'dart:async';

class SplashScreen extends StatefulWidget {
  @override
  _SplashScreenState createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> {
  @override
  void initState() {
    super.initState();
    // Simulate a delay before navigating to the home screen
    Timer(Duration(seconds: 2), () {
      Navigator.of(context).pushReplacementNamed('/home');
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: ClipOval(
          child: Container(
            width: 200, // Adjust size as needed
            height: 200, // Adjust size as needed
            decoration: BoxDecoration(
              image: DecorationImage(
                image: AssetImage('assets/splash_image.png'), // Ensure the image exists at this path
                fit: BoxFit.cover, // Cover the entire area of the container
              ),
            ),
            child: Center(
              // Optionally, you can add a loading spinner or text on top of the image
              // child: CircularProgressIndicator(),
              // child: Text('Loading...'),
            ),
          ),
        ),
      ),
    );
  }
}
